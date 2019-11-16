#-*-coding:utf-8-*-

import os

from configs.vgg_config import config as vgg_config
from configs.lightnet_config import config as lightnet_config_075
from configs.lightnet_config_05 import config as lightnet_config_05
from configs.mobilenet_config import config as mb_config

##### the config for different backbone,
config=lightnet_config_05

os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu
config.TRAIN.num_gpu = 1





