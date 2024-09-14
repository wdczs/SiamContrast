import os
import pdb
import argparse
import time
import yaml
import shutil
import logging
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader

import models.multitask_cd_model as models
from utils.scd_dataset import SKIMultiTaskCDDataset
from utils.sampler_utils import set_seed_, TrainIterationSampler, mutitask_cd_collate
from utils.misc import create_dirs, create_logger, AverageMeter, save_state, IterLRScheduler
from utils.cd_metric import batch_compute_metric
from loss_contrast_rs import SCDCLCELoss

parser = argparse.ArgumentParser(description='PyTorch Change Detection Training')
parser.add_argument('--config', default='cfgs/Szada/suc/Szada_bs32_suc.yaml')
parser.add_argument('--resume_opt', action='store_true')

args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(config)
    file_name = os.path.splitext(os.path.basename(args.config))[0]
    cfg.TRAIN.CKPT = os.path.join(cfg.TRAIN.CKPT, file_name)

device_ids = cfg.TRAIN.DEVICE_IDS
torch.cuda.set_device(device_ids[0])
set_seed_(cfg.TRAIN.SEED)

def main():
    create_dirs('{}/events'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/checkpoints'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/results'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/logs'.format(cfg.TRAIN.CKPT))
    logger = create_logger('global_logger', '{}/logs/log.txt'.format(cfg.TRAIN.CKPT))
    logger.info('{}'.format(cfg))
    shutil.copyfile(args.config, os.path.join(cfg.TRAIN.CKPT, args.config.split('/')[-1]))

    model = models.__dict__[cfg.MODEL.type](cfg=cfg)

    logger.info(
        "=> creating model: \n{}".format(model))

    train_dataset = SKIMultiTaskCDDataset(cfg=cfg, mode='train')
    logger.info('train_set num: {}'.format(len(train_dataset)))


    init_lr = cfg.SOLVER.BASE_LR

    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                 momentum=cfg.SOLVER.MOMENTUM,
                                 weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                 nesterov=True if 'NESTEROV' in cfg.SOLVER and cfg.SOLVER.NESTEROV else False)

    logger.info(optimizer)
    latest_iter = -1

    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

    weight = torch.Tensor(cfg.CD_LOSS.WEIGHT).cuda() if 'WEIGHT' in cfg.CD_LOSS and cfg.CD_LOSS.WEIGHT else None
    cd_criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=cfg.CD_LOSS.IGNORE).cuda()
    if cfg.SEG_LOSS.TYPE == 'ce_aux_contrast':
        seg_criterion = SCDCLCELoss(cfg).cuda()
    else:
        weight = torch.Tensor(cfg.SEG_LOSS.WEIGHT).cuda() if 'WEIGHT' in cfg.SEG_LOSS and cfg.SEG_LOSS.WEIGHT else None
        if cfg.SEG_LOSS.IGNORE is None:
            ignore = -100
        else:
            ignore = cfg.SEG_LOSS.IGNORE
        seg_criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore).cuda()

    train_sampler = TrainIterationSampler(dataset=train_dataset, total_iter=cfg.SOLVER.MAX_ITER,
                                          batch_size=train_dataset.batch_size, last_iter=latest_iter)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_dataset.batch_size,
        shuffle=False,
        num_workers=cfg.TRAIN.WORKERS, pin_memory=False, sampler=train_sampler, collate_fn=mutitask_cd_collate)


    lr_scheduler = IterLRScheduler(optimizer, cfg.SOLVER.LR_STEPS, cfg.SOLVER.LR_MULTS, latest_iter=latest_iter)
    train_val(train_loader, model, cd_criterion, seg_criterion, optimizer, lr_scheduler, latest_iter + 1)

def train_val(train_loader, model, cd_criterion, seg_criterion, optimizer, lr_scheduler, start_iter):
    logger = logging.getLogger('global_logger')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    add_losses = AverageMeter()
    f1s = AverageMeter()
    pccs = AverageMeter()
    kappas = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    # switch to train mode
    end = time.time()
    model.train()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_A = data[0].cuda()
        input_B = data[1].cuda()
        target_cd = data[2].long().cuda()
        target_seg1 = data[3].long().cuda()
        target_seg2 = data[4].long().cuda()
        curr_step = start_iter + i
        lr_scheduler.step(curr_step)
        current_lr = lr_scheduler.get_lr()[0]

        # compute output
        feat_dict, logits_dict = model(input_A, input_B)
        cd_logits, seg1_logits, seg2_logits = logits_dict['cd_logits'], logits_dict['seg1_logits'], logits_dict['seg2_logits']
        cd_loss = cd_criterion(cd_logits, target_cd)


        if cfg.SEG_LOSS.TYPE == 'ce_aux_contrast':
            seg_loss = seg_criterion(feat_dict, logits_dict, target_seg1, target_seg2, target_cd)
        else:
            seg1_loss = seg_criterion(seg1_logits, target_seg1)
            seg2_loss = seg_criterion(seg2_logits, target_seg2)
            seg_loss = seg1_loss + seg2_loss
        loss = cfg.SEG_LOSS.RATIO * seg_loss

        loss += cd_loss

        # set gradient to zero
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update params
        optimizer.step()

        cd_logits = cd_logits.float()
        loss = loss.float()
        batch_val = batch_compute_metric(cfg, cd_logits, data[2].long().cuda())
        losses.update(loss.item())
        add_losses.update(seg_loss.item())

        f1s.update(batch_val['F1'].item())
        precisions.update(batch_val['Pr'].item())
        recalls.update(batch_val['Re'].item())
        pccs.update(batch_val['Pcc'].item())
        kappas.update(batch_val['Kappa'].item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if curr_step % cfg.SOLVER.PRINT_FREQ == 0:
            logger.info('Cfg: {cfg} |'
                        'Iter: [{0}/{1}] |'
                        'Time {batch_time.avg:.3f}({batch_time.val:.3f}) |'
                        'Data {data_time.avg:.3f}({data_time.val:.3f}) |'
                        'Total Loss {total_loss.avg:.4f}({total_loss.val:.4f}) |'
                        'Seg Loss {seg_loss.avg:.4f}({seg_loss.val:.4f}) |'
                        'CD_F1 {f1s.avg:.3f}({f1s.val:.3f}) |'
                        'CD_Pcc {pccs.avg:.3f}({pccs.val:.3f}) |'
                        'CD_Kappa {kappas.avg:.3f}({kappas.val:.3f}) |'
                        'CD_Precision {precisions.avg:.3f}({precisions.val:.3f}) |'
                        'CD_Recall {recalls.avg:.3f}({recalls.val:.3f}) |'
                        'LR {lr:.6f} |'
                        'Total {batch_time.all:.2f}hrs |'.format(
                curr_step, len(train_loader) + start_iter,
                dataset_name=cfg.TRAIN.DATASETS,
                batch_time=batch_time, data_time=data_time,
                total_loss=losses, f1s=f1s, pccs=pccs, kappas=kappas,
                precisions=precisions, recalls=recalls,
                seg_loss=add_losses,
                lr=current_lr, cfg=os.path.basename(args.config)))

        if (curr_step + 1) % cfg.SOLVER.SNAPSHOT == 0:
            ckpt_dict = {
                'step': curr_step + 1,
                'dataset_name': cfg.TRAIN.DATASETS,
                'type': cfg.MODEL.type,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            ckpt_dict['backbone'] = cfg.MODEL.encoder_name
            save_state(ckpt_dict, cfg.TRAIN.CKPT)

if __name__ == '__main__':
    main()