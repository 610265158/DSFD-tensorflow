#-*-coding:utf-8-*-

from easydict import EasyDict as edict

config = edict()

###below are the config params generally used by training
config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.process_num = 5                      ### process_num for data provider
config.TRAIN.prefetch_size = 10                  ### prefect Q size for data provider

config.TRAIN.num_gpu = 1                         ##match with   os.environ["CUDA_VISIBLE_DEVICES"]
config.TRAIN.batch_size = 32                     ###A big batch size may achieve a better result, but the memory is a problem
config.TRAIN.log_interval = 100
config.TRAIN.epoch = 300

config.TRAIN.lr_value_every_epoch = [0.00001,0.0001,0.001,0.0001,0.00001,0.000001]          ####lr policy
config.TRAIN.lr_decay_every_epoch = [1,2,150,200,250]

config.TRAIN.weight_decay_factor = 5.e-4                  ##l2 regular
config.TRAIN.vis=False                                    ##check data flag
config.TRAIN.opt='Adam'
config.TRAIN.mix_precision=True

config.TEST = edict()
config.TEST.score_thres = 0.1
config.TEST.iou_thres = 0.35




config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.num_category=1                                  ###face 1  voc 20 coco 80
config.DATA.num_class = config.DATA.num_category + 1        # +1 background

config.DATA.PIXEL_MEAN = [127., 127., 127.]                 ###rgb
config.DATA.PIXEL_STD = [127., 127., 127.]

config.DATA.hin = 512  # input size
config.DATA.win = 512
config.DATA.max_size=[config.DATA.hin,config.DATA.win]  ##h,w
config.DATA.cover_small_face=10                          ###cover the small faces

config.DATA.mutiscale=True                #if muti scale set False  then config.DATA.MAX_SIZE will be the inputsize
config.DATA.scales=(320,512)


# anchors -------------------------
config.ANCHOR = edict()
config.ANCHOR.rect=True
config.ANCHOR.rect_longer=True       ####    make anchor h/w=1.5
config.ANCHOR.ANCHOR_STRIDE = 16
config.ANCHOR.ANCHOR_SIZES = (32,128, 512)   # sqrtarea of the anchor box
config.ANCHOR.ANCHOR_STRIDES = (8, 16, 32)    # strides for each FPN level. Must be the same length as ANCHOR_SIZES
config.ANCHOR.ANCHOR_RATIOS = (1., 4.) ######           1:2 in size,
config.ANCHOR.POSITIVE_ANCHOR_THRESH = 0.35
config.ANCHOR.NEGATIVE_ANCHOR_THRESH = 0.35
config.ANCHOR.AVG_MATCHES=20
config.ANCHOR.super_match=True

config.MODEL = edict()
config.MODEL.net_structure='Lightnet_0.75'
config.MODEL.model_path = './model/'  # save directory
config.MODEL.pretrained_model=None
config.MODEL.fpn_dims=[128,128,128]
config.MODEL.cpm_dims=[128,128,128]

config.MODEL.fpn=True      ###it's a low level fpn
config.MODEL.cpm=True
config.MODEL.dual_mode=False
config.MODEL.maxout=False
config.MODEL.max_negatives_per_positive= 3.0
config.MODEL.focal_loss=False
