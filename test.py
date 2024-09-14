import os
import sys
import pdb

from skimage import io

import argparse
import time
import yaml
import shutil
import logging
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
from scipy import sparse
import threading

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

import datasets.datasets_catalog as dc
import models.multitask_cd_model as models

from utils.scd_dataset import SKIMultiTaskCDDataset as ChangeDetectionDataset
from utils.sampler_utils import set_seed_, mutitask_cd_collate, TrainIterationSampler
from utils.misc import create_dirs, create_logger, AverageMeter, load_state
from utils.cd_metric import batch_compute_metric, CDMetricAverageMeter, LCMetricAverageMeter, SCMetricAverageMeter, compute_iou_per_class
from utils.scd_metric_util import get_hist, get_scd_metrics


parser = argparse.ArgumentParser(description='PyTorch Change Detection Training')
parser.add_argument('--config', default='cfgs/Szada/suc/Szada_bs32_suc.yaml')
parser.add_argument('--resume_opt', action='store_true')
parser.add_argument('--latest_flag', action='store_true')

args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(config)
    file_name = os.path.splitext(os.path.basename(args.config))[0]
    cfg.TRAIN.CKPT = os.path.join(cfg.TRAIN.CKPT, file_name)

device_ids = cfg.TEST.DEVICE_IDS
torch.cuda.set_device(device_ids[0])
set_seed_(cfg.TRAIN.SEED)

thread_max_num = threading.Semaphore(4)

def post_process_work(param):
    file_name, masked_pred_t1, masked_pred_t2, masked_target_seg1, masked_target_seg2, batch_sc_IoU,  \
     output_cd, target_cd, eval_iteration, cls_color_map, state = param

    cd_gt_np = (masked_target_seg1 != masked_target_seg2).astype(int)
    if cfg.TEST.SAVE_VIS == True and np.any(cd_gt_np == 1):
    # if cfg.TEST.SAVE_VIS == True:
        save_images = []
        for j, masked_pred, masked_gt in zip([0, 1], [masked_pred_t1, masked_pred_t2],
                                             [masked_target_seg1, masked_target_seg2]):
            if 'INDEX' in cfg.TEST and cfg.TEST.INDEX is True:
                save_img = masked_pred.astype(np.uint8)
            else:
                save_img = label_map_color(cfg, cls_color_map, masked_pred)
            save_images.append(save_img)
        file_name_name, file_name_ext = os.path.splitext(file_name)
        # save_path = '{}/results/vis_{}/pred_{}'.format(cfg.TRAIN.CKPT, eval_iteration, file_name)
        save_path = '{}/results/vis_{}/pred_{}'.format(cfg.TRAIN.CKPT, eval_iteration,
                                                       file_name_name + '.png')

        if os.path.exists(save_path):
            # save_path = '{}/results/vis_{}/pred_{}'.format(cfg.TRAIN.CKPT, eval_iteration, file_name_name+'_ano'+file_name_ext)
            save_path = '{}/results/vis_{}/pred_{}'.format(cfg.TRAIN.CKPT, eval_iteration,
                                                           file_name_name + '_ano' + '.png')
            io.imsave(save_path, np.hstack(save_images), check_contrast=False)
        else:
            io.imsave(save_path, np.hstack(save_images), check_contrast=False)


    # measure accuracy and record loss

    batch_val = batch_compute_metric(cfg, output_cd, target_cd)
    with open('{}/results/patch_results_{}_{}.csv'.format(cfg.TRAIN.CKPT, state, eval_iteration), 'a') as fout:
        patch_msg = '{}, {:.4f}, {:.4f}\r\n'.format(file_name, batch_val['F1'].item(), batch_sc_IoU)
        fout.write(patch_msg)

def main():
    state = cfg.TEST.DATALIST

    create_dirs('{}/events'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/checkpoints'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/results'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/logs'.format(cfg.TRAIN.CKPT))

    logger = create_logger('global_logger', '{}/logs/log.txt'.format(cfg.TRAIN.CKPT))
    logger.info('{}'.format(cfg))
    shutil.copyfile(args.config, os.path.join(cfg.TRAIN.CKPT, args.config.split('/')[-1]))

    test_dataset = ChangeDetectionDataset(cfg=cfg, mode=state)
    logger.info('{}_set num: {}'.format(state, len(test_dataset)))
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE), shuffle=False, drop_last=False,
        num_workers=cfg.TRAIN.WORKERS, pin_memory=False, collate_fn=mutitask_cd_collate)
    data_loader = test_loader
    num_classes = test_dataset.num_classes
    cls_color_map = dc.get_vis_colors(cfg.TRAIN.DATASETS)
    cls_color_map['unchange'] = [0, 0, 0]
    num_classes += 1

    if cfg.TEST.ITER == 'all':
        iterations = list(range(cfg.SOLVER.SNAPSHOT, cfg.SOLVER.MAX_ITER+1, cfg.SOLVER.SNAPSHOT))
    elif cfg.TEST.ITER == 'random' or str(cfg.TEST.ITER).isdigit() is True:
        iterations = [cfg.TEST.ITER]
    elif type(cfg.TEST.ITER) == list:
        iterations = cfg.TEST.ITER
    elif type(cfg.TEST.ITER) == EasyDict:
        start_iter = cfg.TEST.ITER.START
        end_iter = cfg.TEST.ITER.END
        iterations = list(range(start_iter, end_iter+1, cfg.SOLVER.SNAPSHOT))
    else:
        print(type(cfg.TEST.ITER), cfg.TEST.ITER)
        pdb.set_trace()
        raise NotImplementedError()
    for iteration in iterations:
        if os.path.exists('{}/results/patch_results_{}_{}.csv'.format(cfg.TRAIN.CKPT, state, iteration)):
            with open('{}/results/patch_results_{}_{}.csv'.format(cfg.TRAIN.CKPT, state, iteration), 'w') as fd_out:
                fd_out.truncate()
        model = models.__dict__[cfg.MODEL.type](cfg=cfg)

        # logger.info(
        #     "=> creating model: \n{}".format(model))
        load_path = '{}/checkpoints/iter_{}_checkpoint.pth.tar'.format(cfg.TRAIN.CKPT, iteration)
        logger.info('=> loading model: {}\n'.format(load_path))
        eval_iteration = load_state(load_path, model, logger, latest_flag=False, optimizer=None)


        model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

        logger = logging.getLogger('global_logger')
        data_time = AverageMeter()


        test_metric = CDMetricAverageMeter(cfg)

        valid_num_classes = num_classes - 1 # unchange class
        if 'IGNORE' in cfg.SEG_LOSS and cfg.SEG_LOSS.IGNORE == 0 and cfg.CD_LOSS.IGNORE == 0:
            valid_num_classes = valid_num_classes - 1 # ignore class
        sek_num_classes = valid_num_classes**2 + 1

        unchange_id_in_seg = list(cls_color_map.keys()).index('unchange')
        change_id = 2

        test_sc_metric = SCMetricAverageMeter(num_classes=valid_num_classes, semch_num_classes=sek_num_classes, unchange_id_in_seg=unchange_id_in_seg)

        hist = np.zeros((sek_num_classes, sek_num_classes))
        hist_LC = np.zeros((valid_num_classes + 1, valid_num_classes + 1))
        logger.info('iter {}: {} ...'.format(eval_iteration, state))
        if cfg.TEST.SAVE_VIS == True:
            create_dirs('{}/results/vis_{}'.format(cfg.TRAIN.CKPT, eval_iteration))

        model.eval()
        end = time.time()
        pbar = tqdm(total=len(test_dataset))

        index = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                input_A = data[0].cuda()
                input_B = data[1].cuda()
                target_seg1 = data[3].long().cuda()
                target_seg2 = data[4].long().cuda()
                if cfg.MODEL.out_channels == 1:
                    target_cd = data[2].float().cuda().clone()
                else:
                    target_cd = data[2].long().cuda().clone()
                # compute output
                feat_dict, logits_dict = model(input_A, input_B)
                cd_logits, seg1_logits, seg2_logits = logits_dict['cd_logits'], logits_dict['seg1_logits'], logits_dict[
                    'seg2_logits']
                output_cd = cd_logits.float()
                output_t1 = seg1_logits.float()
                output_t2 = seg2_logits.float()

                # inc
                # cd_pred = cd_logits.max(1)[1]
                cd_pred = cd_logits.max(1)[1]

                seg1_pred = seg1_logits.max(1)[1]
                seg2_pred = seg2_logits.max(1)[1]
                batch_m_bcd = (cd_pred == change_id).float()
                batch_m_bcd_gt = (target_cd == change_id).float()

                cd_pred = cd_pred[target_cd != 0]
                if cfg.CD_LOSS.TYPE is not None:
                    cd_pred = cd_pred - 1
                seg1_pred = seg1_pred[target_cd!=0] - 1
                seg2_pred = seg2_pred[target_cd!=0] - 1
                target_seg11 = target_seg1[target_cd != 0] - 1
                target_seg22 = target_seg2[target_cd != 0] - 1
                target_1 = target_cd[target_cd != 0] - 1
                batch_m_pred = valid_num_classes * (seg1_logits.max(1)[1] - 1) + seg2_logits.max(1)[1] - 1 + 1
                batch_m_gt = valid_num_classes * (target_seg1 - 1) + target_seg2 - 1 + 1

                test_sc_metric.update(seg1_pred, seg2_pred, cd_pred, target_1, target_seg11, target_seg22)

                # tsq: semantic consistency metrics
                # pdb.set_trace()
                # batch change consistensy
                # Change area consistency:
                batch_sc_IoU_list = []
                current_batch_size = input_A.shape[0]

                for j in range(current_batch_size):
                    m_pred_sc = batch_m_pred[j]
                    m_gt_sc = batch_m_gt[j]
                    # m_pred_sc = m_pred_sc[target_cd[j] == change_id].view(-1)
                    # m_gt_sc = m_gt_sc[target_cd[j] == change_id].view(-1)
                    m_pred_sc = (m_pred_sc * batch_m_bcd[j]).view(-1)
                    m_gt_sc = (m_gt_sc * batch_m_bcd_gt[j]).view(-1)
                    if (m_pred_sc<0).sum()>=1:
                        pdb.set_trace()
                    if (m_gt_sc<0).sum()>=1:
                        pdb.set_trace()

                    v = np.ones_like(m_pred_sc.cpu().numpy())
                    dense_cm = sparse.coo_matrix((v, (m_gt_sc.cpu().numpy(), m_pred_sc.cpu().numpy())),
                                           shape=(sek_num_classes, sek_num_classes), dtype=np.float32).toarray()
                    batch_sc_IoUs = compute_iou_per_class(dense_cm)
                    batch_sc_valid = (batch_sc_IoUs > 1e-7).sum()
                    if batch_sc_valid == 0.0:
                        batch_sc_IoU = 0.0
                    else:
                        batch_sc_IoU = batch_sc_IoUs.sum() / (len(np.unique(batch_m_gt.cpu().numpy())))
                    batch_sc_IoU_list.append(batch_sc_IoU)

                masked_pred_t1 = output_t1.clone().max(1)[1]
                masked_pred_t2 = output_t2.clone().max(1)[1]
                masked_target_seg1 = target_seg1.clone()
                masked_target_seg2 = target_seg2.clone()
                # pdb.set_trace()

                if cfg.CD_LOSS.TYPE is not None:
                    pred_unchange_ind = cd_logits.max(1)[1]==1
                else:
                    pred_unchange_ind = cd_logits.max(1)[1]==0
                masked_pred_t1[pred_unchange_ind] = unchange_id_in_seg
                masked_pred_t2[pred_unchange_ind] = unchange_id_in_seg
                masked_target_seg1[data[2].long().cuda()==1] = unchange_id_in_seg
                masked_target_seg2[data[2].long().cuda()==1] = unchange_id_in_seg
                masked_pred_t1[data[2].long().cuda()==0] = 0
                masked_pred_t2[data[2].long().cuda()==0] = 0


                cd_pred_np = cd_pred.cpu().numpy()
                cd_gt_np = target_1.cpu().numpy()
                masked_pred_t1 = masked_pred_t1.cpu().numpy()
                masked_pred_t2 = masked_pred_t2.cpu().numpy()
                masked_target_seg1 = masked_target_seg1.cpu().numpy()
                masked_target_seg2 = masked_target_seg2.cpu().numpy()

                score_masked_pred_t1 = masked_pred_t1[data[2].long().squeeze()!=0] - 1
                score_masked_pred_t2 = masked_pred_t2[data[2].long().squeeze()!=0] - 1
                score_masked_target_seg1 = masked_target_seg1[data[2].long().squeeze()!=0] - 1
                score_masked_target_seg2 = masked_target_seg2[data[2].long().squeeze()!=0] - 1
                merge_pred = np.where(cd_pred_np == 1, score_masked_pred_t1 * (num_classes - 2) + score_masked_pred_t2 + 1, 0)
                merge_target = np.where(cd_gt_np == 1, score_masked_target_seg1 * (num_classes - 2) + score_masked_target_seg2 + 1, 0)
                try:
                    hist_batch = get_hist(merge_pred, merge_target, sek_num_classes)
                    hist += hist_batch
                except:
                    pdb.set_trace()
                    print(i + 1)

                test_metric.update(output_cd, data[2].long().cuda())

                with thread_max_num:
                    thread_list = []
                    current_batch_size = input_A.shape[0]
                    for j in range(current_batch_size):
                        file_name = os.path.basename(test_dataset.metas[index + j][2])
                        # if file_name == '10911.png':
                        #     pdb.set_trace()
                        params = file_name, masked_pred_t1[j], masked_pred_t2[j], masked_target_seg1[j], masked_target_seg2[j], batch_sc_IoU_list[j],  \
     output_cd[j].unsqueeze(0), data[2].long().cuda()[j].unsqueeze(0), eval_iteration, cls_color_map, state
                        thread = threading.Thread(target=post_process_work, args=(params,))
                        thread.start()
                        thread_list.append(thread)
                    for thread in thread_list:
                        thread.join()

                index += current_batch_size
                pbar.update(current_batch_size)
            pbar.close()
            Sek, IoU_unchange, IoU_change, kappa_n0 = get_scd_metrics(hist)

            sc_IoU, nc_IoU = test_sc_metric.sc_IoU, test_sc_metric.nc_IoU

            # F1, Kappa, Pcc, Pr, Re,
            # Sek, kappa_n0, IoU_unchange, IoU_change
            # mIoU_bc, mIoU_sc, mIoU_nc
            msg = '{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, , ' \
                  '{:.4f}, {:.4f}, {:.4f}, {:.4f}, , ' \
                  '{:.4f}, {:.4f}, {:.4f}, , '.format(
                eval_iteration, test_metric.F1, test_metric.Kappa, test_metric.Pcc, test_metric.Pr, test_metric.Re,
                Sek, kappa_n0, IoU_unchange, IoU_change,
                test_sc_metric.bc_IoU, sc_IoU, nc_IoU)
            msg += '\r\n'
            logger.info(msg)
            with open('{}/results/results_{}.csv'.format(cfg.TRAIN.CKPT, state), 'a') as fout:
                fout.write(msg)

def label_map_color(cfg, cls_color_map, masked_pred, masked_gt=None):
    cm = np.array(list(cls_color_map.values())).astype(np.uint8)
    color_img = cm[masked_pred]
    return color_img


if __name__ == '__main__':
    main()