import os
import cv2
import pdb
import numpy as np
import random
from skimage import io
import torch
from torch.utils.data import Dataset
from datasets import datasets_catalog as dc
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ToTensor(ToTensorV2):
    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, 'masks': self.apply_to_masks}

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(m, **params) for m in masks]



class SKIMultiTaskCDDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.datasets_name = cfg.TRAIN.DATASETS
        self.mode = mode
        assert dc.contains(self.datasets_name), 'Unknown dataset_name: {}'.format(self.datasets_name)
        self.metas = []
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.num_classes = dc.get_num_classes(self.cfg.TRAIN.DATASETS)

        source_file = dc.get_source_index(self.datasets_name)[self.mode]
        print(source_file)
        prefix = dc.get_prefix(self.datasets_name)
        with open(source_file, 'r') as f:
            lines = f.readlines()
            self.info_num = len(lines[0].strip().split(' '))
            for line in lines:
                pkg = line.strip().split(' ')
                for i in range(self.info_num):
                    pkg[i] = os.path.join(prefix, pkg[i])
                self.metas.append(tuple(pkg))
        self.num = len(lines)

        transforms = []
        if self.mode == 'train':
            if self.cfg.AUG.RANDOM_ROTATION == True:
                transforms.append(A.RandomRotate90(True))
            if self.cfg.AUG.RANDOM_HFLIP == True:
                transforms.append(A.HorizontalFlip())

        transforms.extend([
            A.Normalize(mean=dc.get_mean(self.datasets_name)*2,
                      std=dc.get_std(self.datasets_name)*2),
            ToTensor()
        ])
        self.transforms = A.Compose(transforms)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        pkg = []
        for filename in self.metas[idx]:
            pkg.append(io.imread(filename))
        h, w, ch = pkg[0].shape

        # transform: im1, im2, cd_label, label1, label2
        transformed = self.transforms(image=np.concatenate([pkg[0], pkg[1]], axis=2),
                                      masks=[pkg[2], pkg[3], pkg[4]])
        pkg[0], pkg[1] = torch.split(transformed["image"], self.cfg.MODEL.in_channels, dim=0)
        pkg[2], pkg[3], pkg[4] = transformed["masks"]

        return tuple(pkg)
