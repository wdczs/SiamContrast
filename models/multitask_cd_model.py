import pdb
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets.datasets_catalog as dc
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.base import Conv2dReLU, SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.initialization import initialize_decoder, initialize_head

__all__ = ['SiamContrast']

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(True),
            nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        )
    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

class SiamContrast(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        decoder_pyramid_channels = self.cfg.MODEL.decoder_pyramid_channels
        decoder_segmentation_channels = self.cfg.MODEL.decoder_segmentation_channels
        decoder_dropout = self.cfg.MODEL.decoder_dropout
        decoder_merge_policy = self.cfg.MODEL.decoder_merge_policy
        upsampling = self.cfg.MODEL.upsampling

        self.seg_encoder = get_encoder(
            name=self.cfg.MODEL.encoder_name,
            in_channels=self.cfg.MODEL.in_channels,
            depth=self.cfg.MODEL.encoder_depth,
            weights=self.cfg.MODEL.encoder_weights,
        )

        self.seg_decoder = FPNDecoder(
            encoder_channels=self.seg_encoder.out_channels,
            encoder_depth=self.cfg.MODEL.encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        cd_outcs = [outc for outc in self.seg_encoder.out_channels]

        self.proj_head = ProjectionHead(dim_in=list(self.seg_encoder.out_channels)[-1], proj_dim=cfg.SEG_LOSS.proj_dim)

        self.cd_decoder = FPNDecoder(
            encoder_channels=cd_outcs,
            encoder_depth=self.cfg.MODEL.encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )
        self.seg_head = SegmentationHead(
            in_channels=self.seg_decoder.out_channels,
            out_channels=dc.get_num_classes(self.cfg.TRAIN.DATASETS),
            activation=None,
            kernel_size=1,
            upsampling=upsampling,
        )
        self.cd_head = SegmentationHead(
            in_channels=self.cd_decoder.out_channels,
            out_channels=self.cfg.MODEL.out_channels,
            activation=None,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.name = "fpn-{}".format(self.cfg.MODEL.encoder_name)
        initialize_decoder(self.cd_decoder)
        initialize_decoder(self.seg_decoder)
        initialize_head(self.cd_head)
        initialize_head(self.seg_head)

    def forward(self, x1, x2):
        features1 = self.seg_encoder(x1)
        features2 = self.seg_encoder(x2)

        diff_feat_list = []
        for layer_i, (f1, f2) in enumerate(zip(features1, features2)):
            diff = torch.abs(f1 - f2)
            diff_feat_list.append(diff)
        decoder_output1 = self.seg_decoder(*features1)
        decoder_output2 = self.seg_decoder(*features2)
        proj1 = self.proj_head(features1[-1])
        proj2 = self.proj_head(features2[-1])
        decoder_output_cd = self.cd_decoder(*diff_feat_list)

        seg1_logits = self.seg_head(decoder_output1)
        seg2_logits = self.seg_head(decoder_output2)

        cd_logits = self.cd_head(decoder_output_cd)

        feat_dict = {
            'feat1': features1,
            'feat2': features2,
            'proj1': proj1,
            'proj2': proj2,
            'change_feats': decoder_output_cd,
            'decoder_feat1': decoder_output1,
            'decoder_feat2': decoder_output2,
        }
        logits_dict = {
            'cd_logits': cd_logits,
            'seg1_logits': seg1_logits,
            'seg2_logits': seg2_logits,
        }
        return feat_dict, logits_dict

