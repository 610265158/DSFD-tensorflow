#-*-coding:utf-8-*-


import tensorflow as tf
import numpy as np

from lib.core.anchor.box_utils import batch_decode
from lib.core.anchor.nms import batch_non_max_suppression

from lib.core.model.net.resnet.backbone import resnet_ssd
from lib.core.model.net.mobilenet.backbone import mobilenet_ssd
from lib.core.model.net.vgg.backbone import vgg_ssd
from lib.core.model.net.ssd_out import ssd_out
from lib.core.model.net.ssd_loss import ssd_loss
from train_config import config as cfg
from lib.core.anchor.tf_anchors import get_all_anchors_fpn


from lib.helper.logger import logger
def preprocess( image):

    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.DATA.PIXEL_MEAN
        std = np.asarray(cfg.DATA.PIXEL_STD)

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean) * image_invstd                   ###imagenet preprocess just centered the data

    return image
def SSD(images,boxes,labels,L2_reg,training=True):
    images=preprocess(images)

    if 'MobilenetV1' in cfg.MODEL.net_structure:
        ssd_backbne=mobilenet_ssd
    elif 'resnet' in cfg.MODEL.net_structure:
        ssd_backbne = resnet_ssd
    elif 'vgg' in cfg.MODEL.net_structure:
        ssd_backbne = vgg_ssd
    else:
        ssd_backbne=None
        print('a net structure that not supported')

    origin_fms,enhanced_fms=ssd_backbne(images, L2_reg, training)


    print('origin_fms', origin_fms)
    print('enhanced_fms', enhanced_fms)


    with tf.variable_scope('ssd'):

        if not cfg.MODEL.fpn and not cfg.MODEL.dual_mode:

            logger.info('the model was trained as a plain ssd')
            reg_final, cla_final=ssd_out(origin_fms, L2_reg, training)

            reg_loss, cla_loss = ssd_loss(reg_final, cla_final, boxes, labels, 'ohem')
        elif  cfg.MODEL.fpn and not cfg.MODEL.dual_mode:
            logger.info('the model was trained without dual shot')
            reg_final, cla_final = ssd_out(enhanced_fms, L2_reg, training)
            reg_loss, cla_loss = ssd_loss(reg_final, cla_final, boxes, labels, 'ohem')

        elif cfg.MODEL.dual_mode:
            logger.info('the model was trained with dual shot, FEM')
            reg, cla= ssd_out(origin_fms, L2_reg, training,1)
            boxes_small=boxes[:,1::2]
            label_small=labels[:,1::2]

            reg_loss, cla_loss = ssd_loss(reg, cla, boxes_small, label_small, 'ohem')

            with tf.variable_scope('dual'):


                reg_final, cla_final = ssd_out(enhanced_fms, L2_reg, training,1)

                boxes_norm = boxes[:, 0::2]
                label_norm = labels[:, 0::2]

                reg_loss_dual, cla_loss_dual = ssd_loss(reg_final, cla_final, boxes_norm, label_norm,'ohem')


            reg_loss=(reg_loss+reg_loss_dual)
            cla_loss=(cla_loss+cla_loss_dual)


    ###### make it easy to adjust the anchors,      but it trains with a fixed h,w
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    anchors_=get_all_anchors_fpn(max_size=[h,w])

    if cfg.MODEL.dual_mode:
        anchors_ = anchors_[0::2]
    else:
        anchors_ = anchors_
    get_predictions(reg_final,cla_final,anchors_)

    return reg_loss,cla_loss
def get_predictions(box_encodings,cla,anchors, \
                    score_threshold=cfg.TEST.score_thres, \
                    iou_threshold=cfg.TEST.iou_thres,\
                    max_boxes=cfg.TEST.max_detect):
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

        boxes = tf.clip_by_value(boxes, 0.0, 1.0)
        # it has shape [batch_size, num_anchors, 4]
        if 0:
            scores = tf.nn.sigmoid(cla)[:, :, 1:]  ##ignore the bg
        else:
            scores = tf.nn.softmax(cla, axis=2)[:, :, 1:]  ##ignore the bg
        # it has shape [batch_size, num_anchors,class]
        labels = tf.argmax(scores,axis=2)
        # it has shape [batch_size, num_anchors]

        scores = tf.reduce_max(scores,axis=2)
        # it has shape [batch_size, num_anchors]


    with tf.device('/cpu:0'), tf.name_scope('nms'):
        boxes, scores,labels, num_detections = batch_non_max_suppression(
            boxes, scores,labels, score_threshold, iou_threshold, max_boxes
        )

    boxes=tf.identity(boxes,name='boxes')
    scores = tf.identity(scores, name='scores')
    labels = tf.identity(labels, name='labels')
    num_detections = tf.identity(num_detections, name='num_detections')

    return {'boxes': boxes, 'scores': scores, 'num_boxes': num_detections}

