import pdb
import numpy as np
import math
import os


# num_class = 37
# IMAGE_FORMAT = '.png'
# INFER_DIR = './prediction_dir/'
# LABEL_DIR = './label_dir/'

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    fast_hist_result = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    return fast_hist_result


def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa

def get_scd_metrics(hist):
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    print('kappa_n0: {}, math.exp(IoU_fg): {}'.format(kappa_n0, math.exp(IoU_fg)))
    return Sek, iu[0], iu[1], kappa_n0

def get_scd_metrics_unc_last(hist):
    hist_fg = hist[:-1, :-1]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[-1][-1]
    c2hist[0][1] = hist.sum(1)[-1] - hist[-1][-1]
    c2hist[1][0] = hist.sum(0)[-1] - hist[-1][-1]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[-1][-1] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

    print('kappa_n0: {}, math.exp(IoU_fg): {}'.format(kappa_n0, math.exp(IoU_fg)))
    return Sek, iu[0], IoU_fg

