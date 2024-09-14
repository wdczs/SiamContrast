import pdb
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets.datasets_catalog as dc


class PixelContrastLoss(nn.Module):
    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.temperature = self.configer.temperature
        self.base_temperature = self.configer.base_temperature
        self.ignore_label = self.configer.IGNORE
        self.max_samples = self.configer.max_samples
        self.max_views = self.configer.max_views
        self.hard_ratio = 0.5
        if 'hard_ratio' in self.configer:
            self.hard_ratio = self.configer.hard_ratio
        self.drop_one_class = False
        if 'drop_one_class' in self.configer:
            self.drop_one_class = self.configer.drop_one_class

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                num_hard_bd = int(self.hard_ratio * n_view)
                num_easy_bd = n_view - num_hard_bd
                if num_hard >= num_hard_bd and num_easy >= num_easy_bd:
                    num_hard_keep = num_hard_bd
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= num_hard_bd:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= num_easy_bd:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_, cls_weights=None):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if cls_weights is not None:
            cls_weights = torch.Tensor(cls_weights)
            weights = cls_weights[labels_.long()].cuda()
            loss = loss * weights
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None, cls_weights=None):
        if len(labels.shape) == 3:
            labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        # valid_index = labels != self.ignore_label
        # feats_valid, labels_valid = feats[valid_index], labels[valid_index]
        predict = predict.contiguous().view(batch_size, -1)
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        if self.drop_one_class is True and len(torch.unique(labels_)) < 2:
            feats_ = None
        if feats_ is None:
            loss = 0.0
        else:
            loss = self._contrastive(feats_, labels_, cls_weights=cls_weights)
        return loss


class SCDCLCELoss(nn.Module):
    def __init__(self, cfg):
        super(SCDCLCELoss, self).__init__()

        self.cfg = cfg

        self.seg_num_class = dc.get_num_classes(self.cfg.TRAIN.DATASETS)
        ignore_index = cfg.SEG_LOSS.IGNORE
        self.cl_loss_weight = cfg.SEG_LOSS.loss_weight

        self.hard_mode = cfg.SEG_LOSS.hard_mode
        self.mode = cfg.SEG_LOSS.mode
        self.upsample = cfg.SEG_LOSS.upsample

        self.seg_criterion = nn.CrossEntropyLoss(weight=torch.Tensor(cfg.SEG_LOSS.WEIGHT).cuda() if 'WEIGHT' in cfg.SEG_LOSS and cfg.SEG_LOSS.WEIGHT else None, ignore_index=ignore_index)
        self.cd_criterion = nn.CrossEntropyLoss(weight=torch.Tensor(cfg.CD_LOSS.WEIGHT).cuda() if 'WEIGHT' in cfg.CD_LOSS and cfg.CD_LOSS.WEIGHT else None, ignore_index=cfg.CD_LOSS.IGNORE).cuda()
        self.contrast_criterion = PixelContrastLoss(configer=cfg.SEG_LOSS)
        self.crosst_contrast_criterion = PixelContrastLoss(configer=cfg.SEG_LOSS.corsst_params)


    def resize_labels(self, input_labels, h, w):
        labels = deepcopy(input_labels)
        if len(labels.shape) == 3:
            labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (h, w), mode='nearest')
        labels = labels.squeeze(1).long()
        return labels

    def get_predict(self, hard_mode, h_proj, w_proj, cd_pred, pred1, pred2, target_cd, target_seg1, target_seg2):
        if hard_mode == 'pred':
            cd_pred_resize = F.interpolate(input=cd_pred, size=(h_proj, w_proj), mode='bilinear', align_corners=True)
            pred1_resize = F.interpolate(input=pred1, size=(h_proj, w_proj), mode='bilinear', align_corners=True)
            pred2_resize = F.interpolate(input=pred2, size=(h_proj, w_proj), mode='bilinear', align_corners=True)
            _, cd_predict = torch.max(cd_pred_resize, 1)
            _, predict1 = torch.max(pred1_resize, 1)
            _, predict2 = torch.max(pred2_resize, 1)
        elif hard_mode == 'pred_cd':
            cd_pred_resize = F.interpolate(input=cd_pred, size=(h_proj, w_proj), mode='bilinear', align_corners=True)
            pred1_resize = F.interpolate(input=pred1, size=(h_proj, w_proj), mode='bilinear', align_corners=True)
            pred2_resize = F.interpolate(input=pred2, size=(h_proj, w_proj), mode='bilinear', align_corners=True)
            _, cd_predict = torch.max(cd_pred_resize, 1)
            _, predict1 = torch.max(pred1_resize, 1)
            _, predict2 = torch.max(pred2_resize, 1)

            cd_labels_resize = self.resize_labels(target_cd, h_proj, w_proj)
            t1_labels_resize = self.resize_labels(target_seg1, h_proj, w_proj)
            t2_labels_resize = self.resize_labels(target_seg2, h_proj, w_proj)
            if self.cfg.CD_LOSS.IGNORE < 0:
                predict1[cd_labels_resize==0] = t1_labels_resize[cd_labels_resize==0]
                predict2[cd_labels_resize==0] = t2_labels_resize[cd_labels_resize==0]
            elif self.cfg.CD_LOSS.IGNORE == 0:
                predict1[cd_labels_resize==1] = t1_labels_resize[cd_labels_resize==1]
                predict2[cd_labels_resize==1] = t2_labels_resize[cd_labels_resize==1]
            else:
                raise NotImplementedError()

        elif hard_mode is None:
            cd_labels_resize = self.resize_labels(target_cd, h_proj, w_proj)
            t1_labels_resize = self.resize_labels(target_seg1, h_proj, w_proj)
            t2_labels_resize = self.resize_labels(target_seg2, h_proj, w_proj)
            predict1, predict2, cd_predict = t1_labels_resize, t2_labels_resize, cd_labels_resize
        else:
            raise NotImplementedError()
        return predict1, predict2, cd_predict

    def forward(self, feat_dict, logits_dict, target_seg1, target_seg2, target_cd):
        h, w = target_seg1.size(1), target_seg1.size(2)
        proj1, proj2 = feat_dict['proj1'], feat_dict['proj2']
        cd_logits, seg1_logits, seg2_logits = logits_dict['cd_logits'], logits_dict['seg1_logits'], logits_dict['seg2_logits']

        pred1 = seg1_logits
        pred2 = seg2_logits
        cd_pred = cd_logits

        ce_loss_1 = self.seg_criterion(pred1, target_seg1)
        ce_loss_2 = self.seg_criterion(pred2, target_seg2)

        ce_loss = (ce_loss_1 + ce_loss_2) / 2

        if self.upsample is not None:
            h_proj, w_proj = proj1.size(2), proj1.size(3)
            proj1 = F.interpolate(input=proj1, size=(h_proj*self.upsample, w_proj*self.upsample), mode='bilinear', align_corners=True)
            proj2 = F.interpolate(input=proj2, size=(h_proj*self.upsample, w_proj*self.upsample), mode='bilinear', align_corners=True)

        h_proj, w_proj = proj1.size(2), proj1.size(3)

        target_seg1_ = deepcopy(target_seg1)
        target_seg2_ = deepcopy(target_seg2)


        predict1, predict2, cd_predict = self.get_predict(self.cfg.SEG_LOSS.hard_mode, h_proj, w_proj, cd_pred,
                                                          pred1, pred2, target_cd,
                                                          target_seg1, target_seg2)
        loss_contrast1 = self.contrast_criterion(proj1, target_seg1_, predict1)
        loss_contrast2 = self.contrast_criterion(proj2, target_seg2_, predict2)
        loss_contrast_base = (loss_contrast1 + loss_contrast2) / 2

        predict1_c, predict2_c, cd_predict_c = self.get_predict(self.cfg.SEG_LOSS.crosst_hard_mode, h_proj, w_proj, cd_pred,
                                                          pred1, pred2, target_cd,
                                                          target_seg1, target_seg2)
        proj = torch.cat([proj1, proj2], dim=0)
        target_seg = torch.cat([target_seg1_, target_seg2_], dim=0)
        if predict1_c is None or predict2_c is None:
            predict = None
        else:
            predict = torch.cat([predict1_c, predict2_c], dim=0)

        loss_contrast_crosst = self.crosst_contrast_criterion(proj, target_seg, predict)
        loss_contrast = self.cfg.SEG_LOSS.crosstbase2_weight[0] * loss_contrast_base + self.cfg.SEG_LOSS.crosstbase2_weight[1] * loss_contrast_crosst


        loss = ce_loss + self.cl_loss_weight * loss_contrast
        return loss