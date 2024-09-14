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

def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []

        else:
            self.count = 0
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.all = 0.0

    def update(self, val):
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
            self.sum = np.sum(self.history)
        else:
            self.val = val
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count
            self.all = self.sum / 3600

class IterLRScheduler(object):
    def __init__(self, optimizer, milestones, lr_mults, latest_iter=-1):
        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        self.milestones = milestones
        self.lr_mults = lr_mults
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
        self.latest_iter = latest_iter

    def _get_lr(self):
        try:
            pos = self.milestones.index(self.latest_iter)
        except ValueError:
            return list(map(lambda group: group['lr'], self.optimizer.param_groups))
        except:
            raise Exception('wtf?')
        return list(map(lambda group: group['lr']*self.lr_mults[pos], self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.latest_iter + 1
        self.latest_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target_all = target.view(1, -1).expand_as(pred)
    # all
    correct = pred.eq(target_all)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_state(state, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = '{}/checkpoints/iter_{}_checkpoint.pth.tar'.format(save_path, state['step'])
    latest_path = '{}/checkpoints/latest_checkpoint.pth.tar'.format(save_path)
    torch.save(state, model_path)
    shutil.copyfile(model_path, latest_path)


def load_state(path, model, logger=None, latest_flag=True, optimizer=None):
    # pdb.set_trace()
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path) and latest_flag is False:
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' not in checkpoint and 'step' not in checkpoint:
            checkpoint = {'state_dict': checkpoint, 'step': -1}
    elif os.path.isfile(path) and latest_flag is True:
        checkpoint = torch.load(path, map_location='cpu')
    else:
        assert True, "=> no checkpoint found at '{}'".format(path)
    ckpt_keys = set(checkpoint['state_dict'].keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        if logger != None:
            logger.info('caution: missing keys from checkpoint {}: {}'.format(path, k))
        else:
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))

    copying_layers = {}
    ignoring_layers = {}
    for key in own_keys:
        if key not in ckpt_keys:
            continue
        if checkpoint['state_dict'][key].shape == model.state_dict()[key].shape:
            copying_layers[key] = checkpoint['state_dict'][key]
        else:
            ignoring_layers[key] = checkpoint['state_dict'][key]
            if logger != None:
                logger.info('caution: shape mismatched keys from checkpoint {}: {}'.format(path, key))
            else:
                print('caution: shape mismatched keys from checkpoint {}: {}'.format(path, key))


    model.load_state_dict(copying_layers, strict=False)
    eval_iteration = checkpoint['step']
    if logger != None:
        logger.info("=> loaded state from checkpoint '{}' (iter {})".format(path, eval_iteration))
    else:
        print("=> loaded state from checkpoint '{}' (iter {})".format(path, eval_iteration))
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        if logger != None:
            logger.info("=> also loaded optimizer from checkpoint '{}' (iter {})".format(path, eval_iteration))
        else:
            print("=> also loaded optimizer from checkpoint '{}' (iter {})".format(path, eval_iteration))
    return eval_iteration

def param_groups(model):
    conv_weight_group = []
    conv_bias_group = []
    bn_group = []
    feature_weight_group = []
    feature_bias_group = []
    classification_fc_group = []

    normal_group = []
    arranged_names = set()

    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_group.append(m.weight)
            bn_group.append(m.bias)
            arranged_names.add(name + '.weight')
            arranged_names.add(name + '.bias')
        elif isinstance(m, nn.Conv2d):
            conv_weight_group.append(m.weight)
            if m.bias is not None:
                conv_bias_group.append(m.bias)
            arranged_names.add(name + '.weight')
            arranged_names.add(name + '.bias')
        elif isinstance(m, nn.Linear):
            if m.out_features == model.num_classes:
                classification_fc_group.append(m.weight)
                if m.bias is not None:
                    classification_fc_group.append(m.bias)
            else:
                feature_weight_group.append(m.weight)
                if m.bias is not None:
                    feature_bias_group.append(m.bias)

            arranged_names.add(name + '.weight')
            arranged_names.add(name + '.bias')

    for name, param in model.named_parameters():
        if name in arranged_names:
            continue
        else:
            normal_group.append(param)

    return conv_weight_group, conv_bias_group, bn_group, \
        feature_weight_group, feature_bias_group, classification_fc_group, \
           normal_group
