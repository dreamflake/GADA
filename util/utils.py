import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from .verification import evaluate
from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import bcolz
import io
import os
##########################
# 3DDFA_V2
import sys
import argparse
import cv2
import yaml
from _3DDFA_V2.utils.functions import plot_image
from _3DDFA_V2.TDDFA import TDDFA
from _3DDFA_V2.utils.render import render
from _3DDFA_V2.utils.depth import depth
from _3DDFA_V2.utils.pncc import pncc
from _3DDFA_V2.utils.uv import uv_tex
from _3DDFA_V2.utils.pose import viz_pose
from _3DDFA_V2.utils.serialization import ser_to_ply, ser_to_obj
from _3DDFA_V2.utils.functions import draw_landmarks, get_suffix


import logging
################

def master_seed(seed=1234, set_random=True, set_numpy=True, set_tensorflow=False, set_mxnet=False, set_torch=True):

    """
    Set the seed for all random number generators used in the library. This ensures experiments reproducibility and
    stable testing.

    :param seed: The value to be seeded in the random number generators.
    :type seed: `int`
    :param set_random: The flag to set seed for `random`.
    :type set_random: `bool`
    :param set_numpy: The flag to set seed for `numpy`.
    :type set_numpy: `bool`
    :param set_tensorflow: The flag to set seed for `tensorflow`.
    :type set_tensorflow: `bool`
    :param set_mxnet: The flag to set seed for `mxnet`.
    :type set_mxnet: `bool`
    :param set_torch: The flag to set seed for `torch`.
    :type set_torch: `bool`
    """
    import numbers

    logger = logging.getLogger(__name__)
    if not isinstance(seed, numbers.Integral):
        raise TypeError("The seed for random number generators has to be an integer.")

    # Set Python seed
    if set_random:
        import random

        random.seed(seed)

    # Set Numpy seed
    if set_numpy:
        np.random.seed(seed)
        np.random.RandomState(seed)

    # Now try to set seed for all specific frameworks
    if set_tensorflow:
        try:
            import tensorflow as tf

            logger.info("Setting random seed for TensorFlow.")
            if tf.__version__[0] == "2":
                tf.random.set_seed(seed)
            else:
                tf.set_random_seed(seed)
        except ImportError:
            logger.info("Could not set random seed for TensorFlow.")

    if set_mxnet:
        try:
            import mxnet as mx

            logger.info("Setting random seed for MXNet.")
            mx.random.seed(seed)
        except ImportError:
            logger.info("Could not set random seed for MXNet.")

    if set_torch:
        try:
            logger.info("Setting random seed for PyTorch.")
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            logger.info("Could not set random seed for PyTorch.")
def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
    issame = np.load('{}/{}_list.npy'.format(path, name))
    # print(np.shape(issame))
    return carray, issame


def get_val_data(data_path):
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    calfw, calfw_issame = get_val_pair(data_path, 'calfw')
    cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')

    return lfw, cfp_fp, agedb_30, cplfw, calfw, lfw_issame, cfp_fp_issame, agedb_30_issame, cplfw_issame, calfw_issame

def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

    # print(optimizer)


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)


def de_preprocess(tensor):

    return tensor * 0.5 + 0.5





def hflip_batch(imgs_tensor,hflip):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs






def perform_val_with_threshold(embedding_size, batch_size, backbone, carray, issame, nrof_folds = 10, threshold=1.0,tta = False, args=None):


    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    from config import configurations
    face_cfg = configurations[args.model]
    RGB_MEAN = face_cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = face_cfg['RGB_STD']
    THRESHOLD = threshold
    hflip = transforms.Compose([
        de_preprocess,
        transforms.ToPILImage(),
        transforms.functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])
    ccrop = transforms.Compose([
        de_preprocess,
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        from _3DDFA_V2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from _3DDFA_V2.TDDFA_ONNX import TDDFA_ONNX

        #face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = False
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        #face_boxes = FaceBoxes()


    backbone.eval() # switch to evaluation mode
    # Evaluation: LFW
    # Acc: 0.9965, CPLFW
    # Acc: 0.9035
    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size]) 
            if tta:
                ccropped = ccrop_batch(batch,ccrop)
                fliped = hflip_batch(ccropped,hflip )
                emb_batch = backbone(ccropped.cuda())[0].cpu() + backbone(fliped.cuda())[0].cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:

                ccropped = ccrop_batch(batch,ccrop)
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.cuda())[0]).cpu()


            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:]) #[:, [2, 1, 0], :, :]
            if tta:
                ccropped = ccrop_batch(batch,ccrop)
                fliped = hflip_batch(ccropped,hflip)
                emb_batch = backbone(ccropped.cuda())[0].cpu() + backbone(fliped.cuda())[0].cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch,ccrop)
                embeddings[idx:] = l2_norm(backbone(ccropped.cuda())[0]).cpu()



    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    predict_issame = np.less(dist, THRESHOLD)
    tp = np.sum(np.logical_and(predict_issame, issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), issame))
    acc = float(tp + tn) / dist.size

    # tpr, fpr, accuracy, best_thresholds, bad_case = evaluate(embeddings, issame, nrof_folds)
    # buf = gen_plot(fpr, tpr)
    # roc_curve = Image.open(buf)
    # roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return acc,np.logical_and(predict_issame, issame),np.logical_and(np.logical_not(predict_issame), np.logical_not(issame))


def perform_val(embedding_size, batch_size, backbone, carray, issame, nrof_folds = 10, tta = False, args=None):


    # cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    from config import configurations
    face_cfg = configurations[3]
    RGB_MEAN = face_cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = face_cfg['RGB_STD']
    hflip = transforms.Compose([
        de_preprocess,
        transforms.ToPILImage(),
        transforms.functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])
    ccrop = transforms.Compose([
        de_preprocess,
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])




    backbone.eval() # switch to evaluation mode

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size].detach().clone()) #[:, [2, 1, 0], :, :]

            # overlap2 = np.array(batch[0]).transpose(1, 2, 0) * 255.
            # overlap2 = overlap2.astype(np.uint8)
            # plot_image(overlap2)
            if tta:
                ccropped = ccrop_batch(batch,ccrop)
                fliped = hflip_batch(ccropped,hflip )
                emb_batch = backbone(ccropped.cuda())[0].cpu() + backbone(fliped.cuda())[0].cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch,ccrop)
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.cuda())[0]).cpu()
    
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:].detach().clone())#[:, [2, 1, 0], :, :]
            if tta:
                ccropped = ccrop_batch(batch,ccrop)
                fliped = hflip_batch(ccropped,hflip)
                emb_batch = backbone(ccropped.cuda())[0].cpu() + backbone(fliped.cuda())[0].cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch,ccrop)
                embeddings[idx:] = l2_norm(backbone(ccropped.cuda())[0]).cpu()

    tpr, fpr, accuracy, best_thresholds, bad_case = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf



def ccrop_batch(imgs_tensor ,ccrop):
    ccropped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
