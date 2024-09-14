import os
import logging
import shutil
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import pdb
import re
import torch.nn.functional as F
from segmentation_models_pytorch.utils.functional import _take_channels
from scipy import sparse

eps = 1e-7

def batch_compute_metric(cfg, output, target):
    if output.size()[1] == 1:
        p = torch.sigmoid(output)
    else:
        p = torch.softmax(output, 1)[:, -1, :, :]
    p[p>=0.5] = 1.0
    p[p<0.5] = 0.0
    y_pred = p
    y_true = target.float()
    if len(y_pred.size()) == 3:
        y_pred = y_pred.unsqueeze(1)
    if len(y_true.size()) == 3:
        y_true = y_true.unsqueeze(1)
    assert y_pred.size() == y_true.size()
    if 'CD_LOSS' in cfg and cfg.CD_LOSS.IGNORE == 0:
        fore_index = y_true != cfg.CD_LOSS.IGNORE
        y_pred, y_true = y_pred[fore_index], y_true[fore_index]
        y_true[y_true==1] = 0.0
        y_true[y_true==2] = 1.0
    if 'MAIN_LOSS' in cfg and cfg.MAIN_LOSS.IGNORE == 0:
        fore_index = y_true != cfg.MAIN_LOSS.IGNORE
        y_pred, y_true = y_pred[fore_index], y_true[fore_index]
        y_true[y_true==1] = 0.0
        y_true[y_true==2] = 1.0
    TP = ((y_pred.long() & y_true.long())).sum()
    TN = ((1 - y_pred) * (1 - y_true)).sum()
    FP = (y_pred * (1 - y_true)).sum()
    FN = ((1 - y_pred) * y_true).sum()

    N = TP + TN + FP + FN

    Pcc = (TP + TN) / N
    PRE = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (N * N)
    Kappa = (Pcc - PRE) / (1 - PRE) if 1-PRE else N-N
    Pr = TP / (TP + FP) if TP+FP else N-N
    Re = TP / (TP + FN) if TP+FN else N-N
    F1 = 2 * Pr * Re / (Pr + Re) if Pr+Re else N-N

    keys = ['TP', 'TN', 'FP', 'FN', 'Pcc', 'Kappa', 'Pr', 'Re', 'F1']
    values = [TP, TN, FP, FN, Pcc, Kappa, Pr, Re, F1]

    return dict(zip(keys, values))

class CDMetricAverageMeter():
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def update(self, preds, target):
        if preds.size()[1] == 1:
            preds = torch.sigmoid(preds)
        else:
            preds = torch.softmax(preds, 1)[:, -1, :, :]
        preds, target = preds.float(), target.float()
        preds[preds >= 0.5] = 1.0
        preds[preds < 0.5] = 0.0
        y_pred = preds.unsqueeze(1) if len(preds.size()) == 3 else preds
        y_true = target.unsqueeze(1) if len(target.size()) == 3 else target

        if 'CD_LOSS' in self.cfg and self.cfg.CD_LOSS.IGNORE == 0:
            fore_index = y_true != self.cfg.CD_LOSS.IGNORE
            y_pred, y_true = y_pred[fore_index], y_true[fore_index]
            y_true[y_true == 1] = 0.0  # This should be done before y_true[y_true==change] = 1.0
            y_true[y_true == 2] = 1.0
        if 'MAIN_LOSS' in self.cfg and self.cfg.MAIN_LOSS.IGNORE == 0:
            fore_index = y_true != self.cfg.MAIN_LOSS.IGNORE
            y_pred, y_true = y_pred[fore_index], y_true[fore_index]
            y_true[y_true == 1] = 0.0  # This should be done before y_true[y_true==change] = 1.0
            y_true[y_true == 2] = 1.0

        self.TP += ((y_pred.long() & y_true.long())).sum()
        self.TN += ((1 - y_pred) * (1 - y_true)).sum()
        self.FP += (y_pred * (1 - y_true)).sum()
        self.FN += ((1 - y_pred) * y_true).sum()
        self.N = self.TP + self.TN + self.FP + self.FN

        Pcc = (self.TP + self.TN) / self.N
        PRE = ((self.TP + self.FP) * (self.TP + self.FN) + (self.FN + self.TN) * (self.FP + self.TN)) / (self.N * self.N)
        Kappa = (Pcc - PRE) / (1 - PRE) if 1 - PRE else self.N - self.N
        Pr = self.TP / (self.TP + self.FP) if self.TP + self.FP else self.N - self.N
        Re = self.TP / (self.TP + self.FN) if self.TP + self.FN else self.N - self.N
        F1 = 2 * Pr * Re / (Pr + Re) if Pr + Re else self.N - self.N
        self.IoU_unchange = self.TN/(self.TN+self.FP+self.FN)
        self.IoU_change = self.TP/(self.TP+self.FP+self.FN)

        self.Pcc, self.Kappa, self.Pr, self.Re, self.F1 = Pcc, Kappa, Pr, Re, F1

    def reset(self):
        self.TP = 0.
        self.TN = 0.
        self.FP = 0.
        self.FN = 0.
        self.N = 0.

def cohen_kappa_score(cm_th):
    cm_th = cm_th.astype(np.float32)
    n_classes = cm_th.shape[0]
    sum0 = cm_th.sum(axis=0)
    sum1 = cm_th.sum(axis=1)
    expected = np.outer(sum0, sum1) / (np.sum(sum0) + eps)
    w_mat = np.ones([n_classes, n_classes])
    w_mat.flat[:: n_classes + 1] = 0
    k = np.sum(w_mat * cm_th) / (np.sum(w_mat * expected) + eps)
    return 1. - k

def compute_iou_per_class(confusion_matrix):
    """
    Args:
        confusion_matrix: numpy array [num_classes, num_classes] row - gt, col - pred
    Returns:
        iou_per_class: float32 [num_classes, ]
    """
    sum_over_row = np.sum(confusion_matrix, axis=0)
    sum_over_col = np.sum(confusion_matrix, axis=1)
    diag = np.diag(confusion_matrix)
    denominator = sum_over_row + sum_over_col - diag

    iou_per_class = diag / (denominator+1e-7)

    return iou_per_class

def lc_batch_compute_metric(output, target, binary=False, ignore_channels=None):
    # generate one-hot gt
    num_classes = output.shape[1]
    _total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)
    output = output.max(1)[1]
    if ignore_channels[0] is not None:
        output = output[target != ignore_channels[0]]
        target = target[target != ignore_channels[0]]
    else:
        output = output.view(-1)
        target = target.view(-1)
    v = np.ones_like(output.cpu().numpy())
    cm = sparse.coo_matrix((v, (target.cpu().numpy(), output.cpu().numpy())), shape=(num_classes, num_classes), dtype=np.float32)
    _total += cm
    dense_cm = _total.toarray()
    IoUs = compute_iou_per_class(dense_cm)

    output_onehot = torch.nn.functional.one_hot(output, num_classes)
    target_onehot = torch.nn.functional.one_hot(target, num_classes)
    pr, gt = _take_channels(output_onehot, target_onehot, ignore_channels=ignore_channels)
    if binary is True:
        pr, gt = pr[:, -1], gt[:, -1]

    TP = torch.sum(pr * gt).float()
    FP = torch.sum(pr * (1- gt)).float()
    FN = torch.sum((1 - pr) * gt).float()
    TN = torch.sum((1 - pr) * (1- gt)).float()
    N = TP + TN + FP + FN

    Pcc = (TP + TN) / N
    PRE = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (N * N)
    Kappa = (Pcc - PRE + eps) / (1 - PRE + eps)
    Pr = (TP + eps) / (TP + FP + eps)
    Re = (TP + eps) / (TP + FN + eps)
    F1 = (2 * Pr * Re + eps) / (Pr + Re + eps)
    if ignore_channels[0] is None:
        IoU = IoUs.sum() / num_classes
    else:
        IoU = (IoUs.sum()-IoUs[ignore_channels[0]]) / (num_classes-len(ignore_channels))
    keys = ['TP', 'TN', 'FP', 'FN', 'Pcc', 'Kappa', 'Pr', 'Re', 'F1', 'IoU', 'IoUs']
    values = [TP, TN, FP, FN, Pcc, Kappa, Pr, Re, F1, IoU, IoUs]
    return dict(zip(keys, values))

class LCMetricAverageMeter(object):
    def __init__(self, binary, num_classes, ignore_channels=None):
        self.num_classes = num_classes
        self.ignore_channels = ignore_channels
        self.binary = binary
        self.reset()

    def update(self, output, target, cd_logits=None):
        # generate one-hot gt
        num_classes = output.shape[1]
        output = output.max(1)[1]
        if cd_logits is not None:
            if cd_logits.size()[1] == 1:
                cd_pred_ind = torch.sigmoid(cd_logits).squeeze()
                cd_pred_ind[cd_pred_ind >= 0.5] = True
                cd_pred_ind[cd_pred_ind < 0.5] = False
                cd_pred_ind = cd_pred_ind.bool()
            else:
                cd_pred_ind = cd_logits[:, -1, :, :] >= cd_logits[:, -2, :, :]
            cd_pred_ind = cd_pred_ind.repeat(2, 1, 1)
            # pdb.set_trace()
            output = output[cd_pred_ind]
            target = target[cd_pred_ind]

        if self.ignore_channels[0] is not None:
            output = output[target != self.ignore_channels[0]]
            target = target[target != self.ignore_channels[0]]
        else:
            output = output.view(-1)
            target = target.view(-1)
        v = np.ones_like(output.cpu().numpy())
        cm = sparse.coo_matrix((v, (target.cpu().numpy(), output.cpu().numpy())), shape=(num_classes, num_classes), dtype=np.float32)
        self._total += cm
        dense_cm = self._total.toarray()
        self.IoUs = compute_iou_per_class(dense_cm)
        if self.ignore_channels[0] == 0:
            self.Kappa = cohen_kappa_score(dense_cm[1:, 1:])
        elif self.ignore_channels[0] > 0:
            raise NotImplementedError()
        else:
            self.Kappa = cohen_kappa_score(dense_cm)
        output_onehot = torch.nn.functional.one_hot(output, num_classes)
        target_onehot = torch.nn.functional.one_hot(target, num_classes)
        pr, gt = _take_channels(output_onehot, target_onehot, ignore_channels=self.ignore_channels)
        if self.binary is True:
            pr, gt = pr[:, -1], gt[:, -1]
        self.TP += torch.sum(pr * gt)
        self.FP += torch.sum(pr * (1 - gt))
        self.FN += torch.sum((1 - pr) * gt)
        self.TN += torch.sum((1 - pr) * (1 - gt))
        self.N = self.TP + self.TN + self.FP + self.FN

        self.Pcc = (self.TP + self.TN) / self.N
        PRE = ((self.TP + self.FP) * (self.TP + self.FN) + (self.FN + self.TN) * (self.FP + self.TN)) / (self.N * self.N)
        self.Pr = (self.TP + eps) / (self.TP + self.FP + eps)
        self.Re = (self.TP + eps) / (self.TP + self.FN + eps)
        self.F1 = (2 * self.Pr * self.Re + eps) / (self.Pr + self.Re + eps)
        if self.ignore_channels[0] is None:
            self.IoU = self.IoUs.sum() / self.num_classes
        else:
            self.IoU = (self.IoUs.sum() - self.IoUs[self.ignore_channels[0]]) / (self.num_classes - len(self.ignore_channels))

    def reset(self):
        self.TP = 0.
        self.TN = 0.
        self.FP = 0.
        self.FN = 0.
        self.N = 0.
        self._total = sparse.coo_matrix((self.num_classes, self.num_classes), dtype=np.float32)

    def save_cm(self, path):
        np.save(path, self._total.toarray().astype(float))

# tsq: semantic consistency metrics
class SCMetricAverageMeter(object):
    def __init__(self, num_classes, semch_num_classes, unchange_id_in_seg):
        self.num_classes = num_classes
        self.semch_num_classes = semch_num_classes
        self.unchange_id_in_seg = unchange_id_in_seg
        self.valid_land_cover_num_classes = set(list(range(num_classes)))
        self.valid_change_num_classes = set([0, 1])
        self.valid_semantic_change_num_classes = set(list(range(semch_num_classes)))
        self.change_from_to_set = set()
        self.no_change_from_to_set = set()
        self.reset()

    def update(self, seg1_pred, seg2_pred, bcd_pred, target_1, target_seg11, target_seg22):
        unique_set = set(np.unique(seg1_pred.cpu().numpy()))
        assert unique_set.issubset(self.valid_land_cover_num_classes), "unrecognized land-cover pred1 number"
        unique_set = set(np.unique(target_seg11[target_seg11!=(self.unchange_id_in_seg-1)].cpu().numpy()))
        assert unique_set.issubset(self.valid_land_cover_num_classes), "unrecognized land-cover label1 number"
        ###################
        unique_set = set(np.unique(seg2_pred.cpu().numpy()))
        assert unique_set.issubset(self.valid_land_cover_num_classes), "unrecognized land-cover pred2 number"
        unique_set = set(np.unique(target_seg22[target_seg22!=(self.unchange_id_in_seg-1)].cpu().numpy()))
        assert unique_set.issubset(self.valid_land_cover_num_classes), "unrecognized land-cover label2 number"
        ###################
        unique_set = set(np.unique(bcd_pred.cpu().numpy()))
        assert unique_set.issubset(self.valid_change_num_classes), "unrecognized bcd pred number"
        unique_set = set(np.unique(target_1.cpu().numpy()))
        assert unique_set.issubset(self.valid_change_num_classes), "unrecognized bcd label number"

        # Bi area consistency:
        self.updata_binary(seg1_pred, seg2_pred, bcd_pred)

        m_pred = self.num_classes * seg1_pred + seg2_pred + 1
        m_gt = self.num_classes * target_seg11 + target_seg22 + 1

        # Change area consistency:
        self.m_pred_sc = m_pred * bcd_pred
        self.m_gt_sc = m_gt * target_1

        unique_set = set(np.unique(self.m_pred_sc.cpu().numpy()))
        assert unique_set.issubset(self.valid_semantic_change_num_classes), "unrecognized semantic change pred number"
        unique_set = set(np.unique(self.m_gt_sc.cpu().numpy()))
        assert unique_set.issubset(self.valid_semantic_change_num_classes), "unrecognized semantic change label number"
        self.update_sc(self.m_pred_sc, self.m_gt_sc)
        change_from_to_labels, change_counts = np.unique(self.m_gt_sc.cpu().numpy(), return_counts=True)
        self.change_from_to_set.update(change_from_to_labels)

        # No-change area consistency:
        if self.unchange_id_in_seg == 0:
            self.nc_IoUs = 0.0
            self.nc_IoU = 0.0
        else:
            self.m_pred_nc = m_pred * (1-bcd_pred)
            self.m_gt_nc = m_gt * (1-target_1)
            unique_set = set(np.unique(self.m_pred_nc.cpu().numpy()))
            assert unique_set.issubset(self.valid_semantic_change_num_classes), "unrecognized semantic change pred number"
            unique_set = set(np.unique(self.m_gt_nc.cpu().numpy()))
            assert unique_set.issubset(self.valid_semantic_change_num_classes), "unrecognized semantic change label number"
            self.update_nc(self.m_pred_nc, self.m_gt_nc)
            nochange_from_to_labels, nochange_counts = np.unique(self.m_gt_nc.cpu().numpy(), return_counts=True)
            self.no_change_from_to_set.update(nochange_from_to_labels)

    def updata_binary(self, seg1_pred, seg2_pred, target_1):
        output = (seg1_pred != seg2_pred).float()
        target = target_1.float()
        output = output.view(-1)
        target = target.view(-1)
        v = np.ones_like(output.cpu().numpy())
        cm = sparse.coo_matrix((v, (target.cpu().numpy(), output.cpu().numpy())), shape=(2, 2), dtype=np.float32)
        self._total_pcc_binary += cm
        dense_cm = self._total_pcc_binary.toarray()
        self.bc_IoUs = compute_iou_per_class(dense_cm)
        self.bc_IoU = self.bc_IoUs.sum() / (self.bc_IoUs > 1e-7).sum()

    def update_sc(self, output, target):
        # generate one-hot gt
        output = output.view(-1)
        target = target.view(-1)
        v = np.ones_like(output.cpu().numpy())
        cm = sparse.coo_matrix((v, (target.cpu().numpy(), output.cpu().numpy())), shape=(self.semch_num_classes, self.semch_num_classes), dtype=np.float32)
        self._total_sc += cm
        dense_cm = self._total_sc.toarray()
        self.sc_IoUs = compute_iou_per_class(dense_cm)
        self.sc_IoU = self.sc_IoUs.sum() / (len(self.change_from_to_set) + 1e-7)
    def update_nc(self, output, target):
        # generate one-hot gt
        output = output.view(-1)
        target = target.view(-1)
        v = np.ones_like(output.cpu().numpy())
        try:
            cm = sparse.coo_matrix((v, (target.cpu().numpy(), output.cpu().numpy())), shape=(self.semch_num_classes, self.semch_num_classes), dtype=np.float32)
        except:
            pdb.set_trace()
        self._total_nc += cm
        dense_cm = self._total_nc.toarray()
        self.nc_IoUs = compute_iou_per_class(dense_cm)
        self.nc_IoU = self.nc_IoUs.sum() / (len(self.no_change_from_to_set) + 1e-7)

    def reset(self):
        self.TP = 0.
        self.TN = 0.
        self.FP = 0.
        self.FN = 0.
        self.N = 0.
        self._total_pcc_binary = sparse.coo_matrix((2, 2), dtype=np.float32)
        self._total_sc = sparse.coo_matrix((self.semch_num_classes, self.semch_num_classes), dtype=np.float32)
        self._total_nc = sparse.coo_matrix((self.semch_num_classes, self.semch_num_classes), dtype=np.float32)

    def save_bcd_pcc_cm(self, path):
        np.save(path, self._total_pcc_binary.toarray().astype(float))
    def save_sc_cm(self, path):
        np.save(path, self._total_sc.toarray().astype(float))
    def save_nc_cm(self, path):
        np.save(path, self._total_nc.toarray().astype(float))