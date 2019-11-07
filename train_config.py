#-*-coding:utf-8-*-

import os

from configs.vgg_config import config as vgg_config
from configs.lightnet_config import config as lightnet_config
from configs.mobilenet_config import config as mb_config
##### the config for different backbone,
config=lightnet_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu
config.TRAIN.num_gpu = 1





