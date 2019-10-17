#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

from configs.vgg_config import config as vgg_config
from configs.SF_config import config as shufflenet_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu


##### the config for different backbone
config=vgg_config

###below are the config params generally used by training
config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.process_num = 5                      ### process_num for data provider
config.TRAIN.prefetch_size = 10                  ### prefect Q size for data provider

config.TRAIN.num_gpu = 1                         ##match with   os.environ["CUDA_VISIBLE_DEVICES"]
config.TRAIN.batch_size = 8                    ###A big batch size may achieve a better result, but the memory is a problem
config.TRAIN.log_interval = 100
config.TRAIN.epoch = 300                      ###just keep training , evaluation shoule be take care by yourself,
                                               ### generally 10,0000 iters is enough

config.TRAIN.lr_value_every_epoch = [0.000001,0.00001,0.0001,0.00001,0.000001,0.0000001]          ####lr policy
config.TRAIN.lr_decay_every_epoch = [1,2,40,50,60]

config.TRAIN.weight_decay_factor = 5.e-4                  ##l2 regular
config.TRAIN.vis=False                                    ##check data flag
config.TRAIN.opt='Adam'
config.TRAIN.mix_precision=True

config.TEST = edict()
config.TEST.parallel_iterations=8
config.TEST.score_thres = 0.05
config.TEST.iou_thres = 0.3
config.TEST.max_detect = 1500



config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.num_category=1                                  ###face 1  voc 20 coco 80
config.DATA.num_class = config.DATA.num_category + 1        # +1 background

config.DATA.PIXEL_MEAN = [123., 116., 103.]                 ###rgb
config.DATA.PIXEL_STD = [58., 57., 57.]

config.DATA.hin = 512  # input size
config.DATA.win = 512
config.DATA.max_size=[config.DATA.hin,config.DATA.win]  ##h,w
config.DATA.cover_small_face=5                          ###cover the small faces

config.DATA.mutiscale=False                #if muti scale set False  then config.DATA.MAX_SIZE will be the inputsize
config.DATA.scales=(320,640)



from lib.core.anchor.anchor import Anchor

config.ANCHOR.achor=Anchor()



