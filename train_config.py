#-*-coding:utf-8-*-

import os

from configs.vgg_config import config as vgg_config
from configs.SF_config import config as shufflenet_config


##### the config for different backbone
config=shufflenet_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu
config.TRAIN.num_gpu = 1



from lib.core.anchor.anchor import Anchor
config.ANCHOR.achor=Anchor()



