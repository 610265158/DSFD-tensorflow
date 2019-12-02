#-*-coding:utf-8-*-


import tensorflow as tf
import numpy as np

from lib.core.anchor.box_utils import batch_decode


from lib.core.model.net.light.lightnet import shufflenet_v2,shufflenet_v2_fpn_cpm

from lib.core.model.net.ssd_loss import ssd_loss
from train_config import config as cfg
from lib.core.anchor.tf_anchors import get_all_anchors_fpn


from lib.helper.logger import logger

from lib.core.model.net.vgg.ssd_head import SSDHead

from lib.core.anchor.anchor import anchor_tools

class DSFD():

    def __init__(self,):
        self.ssd_backbone = shufflenet_v2_fpn_cpm  ### it is a func
        #self.ssd_backbone=shufflenet_v2                 ### it is a func
        self.ssd_head=SSDHead()                         ### it is a class


    def forward(self,inputs,boxes,labels,l2_regulation,training_flag,with_loss=True):

        ###preprocess
        inputs=self.preprocess(inputs)

        ### extract feature maps
        origin_fms=self.ssd_backbone(inputs,training_flag)


        print(origin_fms)
        ### head, regresssion and class

        #### train as a dsfd  , anchor with 1 ratios per pixel ,   two shot
        logger.info('train with dsfd ')

        reg, cls = self.ssd_head(origin_fms, l2_regulation, training_flag,ratios_per_pixel=2)

        ### calculate loss


        reg_loss, cls_loss = ssd_loss(reg, cls, boxes, labels, 'ohem')






        ###### adjust the anchors to the image shape, but it trains with a fixed h,w

        anchors_=anchor_tools.anchors
        anchors_[:, 0] = anchors_[:, 0] / cfg.DATA.win
        anchors_[:, 1] = anchors_[:, 1] / cfg.DATA.hin
        anchors_[:, 2] = anchors_[:, 2] / cfg.DATA.win
        anchors_[:, 3] = anchors_[:, 3] / cfg.DATA.hin

        self.postprocess(reg, cls, anchors_)


        return reg_loss,cls_loss

    def preprocess(self,image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)

            mean = cfg.DATA.PIXEL_MEAN
            std = np.asarray(cfg.DATA.PIXEL_STD)

            image_mean = tf.constant(mean, dtype=tf.float32)
            image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
            image = (image - image_mean) * image_invstd  ###imagenet preprocess just centered the data

        return image

    def postprocess(self,box_encodings,cls,anchors):
        """Postprocess outputs of the network.

        Returns:
            boxes: a float tensor with shape [batch_size, N, 4].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.

            where N = max_boxes.
        """
        with tf.name_scope('postprocessing'):
            boxes = batch_decode(box_encodings, anchors)
            # if the images were padded we need to rescale predicted boxes:


            # it has shape [batch_size, num_anchors, 4]
            scores = tf.nn.softmax(cls, axis=2)[:, :, 1:]  ##ignore the bg

            # it has shape [batch_size, num_anchors,class]
            labels = tf.argmax(scores, axis=2)
            # it has shape [batch_size, num_anchors]

            scores = tf.reduce_max(scores, axis=2)
            # it has shape [batch_size, num_anchors]
            scores = tf.expand_dims(scores, axis=-1)
            # it has shape [batch_size, num_anchors]

        res = tf.concat([boxes, scores], axis=2)
        res=tf.identity(res,name='outputs')
        return res
