#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu



# anchors -------------------------
config.ANCHOR = edict()
config.ANCHOR.rect=True
config.ANCHOR.rect_longer=True       ####    make anchor h/w=1.5
config.ANCHOR.ANCHOR_STRIDE = 16
config.ANCHOR.ANCHOR_SIZES = (16,32,64, 128, 256, 512)   # sqrtarea of the anchor box
config.ANCHOR.ANCHOR_STRIDES = (4, 8,16, 32, 64, 128)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
config.ANCHOR.ANCHOR_RATIOS = (1., 4.) ######           1:2 in size,
config.ANCHOR.POSITIVE_ANCHOR_THRESH = 0.35
config.ANCHOR.NEGATIVE_ANCHOR_THRESH = 0.35
config.ANCHOR.AVG_MATCHES=20
config.ANCHOR.super_match=True





#vgg as basemodel. if vgg, set config.TRAIN.norm ='None', achieves fddb 0.987
config.MODEL = edict()
config.MODEL.net_structure='vgg'
config.MODEL.l2_norm=[10.,8.,5.]
config.MODEL.model_path = './model/'  # save directory
config.MODEL.pretrained_model=None#'vgg_16.ckpt'
config.MODEL.fpn_dims=[256,512,512,1024,512,256]
config.MODEL.fem_dims=512


# config.MODEL = edict()
# config.MODEL.net_structure='shufflenet'
# config.MODEL.model_path = './model/'  # save directory
# config.MODEL.pretrained_model=None#'vgg_16.ckpt'
# config.MODEL.fpn_dims=[48,96,96,96,96,96]
# config.MODEL.fem_dims=96

config.MODEL.fpn=True
config.MODEL.dual_mode=True
config.MODEL.maxout=True
config.MODEL.max_negatives_per_positive= 3.0

