import sys
sys.path.append('.')

import tensorflow as tf
import numpy as np

from lib.core.anchor.box_utils import batch_decode
from lib.core.anchor.nms import batch_non_max_suppression
from lib.core.anchor.tf_anchors import get_all_anchors_fpn

from train_config import config as cfg




def l2_normalization(x, scale):

    x = scale*tf.nn.l2_normalize(x, axis=-1)
    return x




class CPM(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(CPM, self).__init__()

        dim = cfg.MODEL.cpm_dims

        self.eyes1=tf.keras.Sequential([tf.keras.layers.Conv2D(filters=dim//2,
                                                                kernel_size=(3,3),
                                                                dilation_rate=(1, 1),
                                                                padding='same',
                                                                kernel_initializer=kernel_initializer),
                                           tf.keras.layers.ReLU()])



        self.eyes2_1 = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=dim//2,
                                                                kernel_size=(3,3),
                                                                dilation_rate=(2, 2),
                                                                padding='same',
                                                                kernel_initializer=kernel_initializer),
                                           tf.keras.layers.ReLU()])

        self.eyes2 = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=dim//4,
                                                                kernel_size=(3,3),
                                                                dilation_rate=(1, 1),
                                                                padding='same',
                                                                kernel_initializer=kernel_initializer),
                                           tf.keras.layers.ReLU()])


        self.eyes3_1 = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=dim//4,
                                                                kernel_size=(3,3),
                                                                dilation_rate=(2, 2),
                                                                padding='same',
                                                                kernel_initializer=kernel_initializer),
                                           tf.keras.layers.ReLU()])

        self.eyes3 = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=dim//4,
                                                                kernel_size=(3,3),
                                                                dilation_rate=(1, 1),
                                                                padding='same',
                                                                kernel_initializer=kernel_initializer),
                                           tf.keras.layers.ReLU()])




    def call(self, x,training):

        cpm1=self.eyes1(x,training=training)

        cpm2_1=self.eyes2_1(x,training=training)
        cpm2=self.eyes2(cpm2_1,training=training)

        cpm3_1 = self.eyes3_1(cpm2_1,training=training)
        cpm3 =self.eyes3(cpm3_1,training=training)

        return tf.concat([cpm1,cpm2,cpm3],axis=3)


class Fpn(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(Fpn, self).__init__()

        dims_list = cfg.MODEL.fpn_dims

        self.conv_3_2 = tf.keras.layers.Conv2D(filters=dims_list[2],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )
        self.conv_2_2 = tf.keras.layers.Conv2D(filters=dims_list[2],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )


        self.conv_2_1 = tf.keras.layers.Conv2D(filters=dims_list[1],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )
        self.conv_1_1 = tf.keras.layers.Conv2D(filters=dims_list[1],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )

        self.conv_1_0 = tf.keras.layers.Conv2D(filters=dims_list[0],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )
        self.conv_0_0 = tf.keras.layers.Conv2D(filters=dims_list[0],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )

        self.upsample=tf.keras.layers.UpSampling2D()
    def __call__(self, fms,training):


        of0,of1,of2,of3,of4,of5=fms

        upsample=self.conv_3_2(of3)

        lateral=self.conv_2_2(of2)

        fpn2=self.upsample_product(upsample,lateral)

        upsample = self.conv_2_1(fpn2)

        lateral = self.conv_1_1(of1)
        fpn1 = self.upsample_product(upsample, lateral)

        upsample = self.conv_1_0(fpn1)

        lateral = self.conv_0_0(of0)
        fpn0 = self.upsample_product(upsample, lateral)

        return [fpn0,fpn1,fpn2,of3,of4,of5]



    def upsample_product(self,x,y):

        x_upsample=self.upsample(x)
        return x_upsample*y

class Extra(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(Extra, self).__init__()

        self.extra_conv1=tf.keras.Sequential([tf.keras.layers.Conv2D(filters=512,
                                                                     kernel_size=(3, 3),
                                                                     strides=2,
                                                                     padding='same',
                                                                     kernel_initializer=kernel_initializer
                                                                     ),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv2D(filters=1024,
                                                                       kernel_size=(1, 1),
                                                                       padding='same',
                                                                       kernel_initializer=kernel_initializer),
                                                tf.keras.layers.ReLU()
                                                ])

        self.extra_conv2=tf.keras.Sequential([tf.keras.layers.Conv2D(filters=256,
                                                                     kernel_size=(3, 3),
                                                                     padding='same',
                                                                     kernel_initializer=kernel_initializer),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv2D(filters=512,
                                                                       kernel_size=(3, 3),
                                                                       strides=2,
                                                                       padding='same',
                                                                       kernel_initializer=kernel_initializer
                                                                       ),
                                                tf.keras.layers.ReLU()
                                                ])

        self.extra_conv3=tf.keras.Sequential([ tf.keras.layers.Conv2D(filters=128,
                                                                      kernel_size=(3, 3),
                                                                      padding='same',
                                                                      kernel_initializer=kernel_initializer
                                                                      ),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv2D(filters=256,
                                                                       kernel_size=(3, 3),
                                                                       strides=2,
                                                                       padding='same',
                                                                       kernel_initializer=kernel_initializer
                                                                       ),
                                                tf.keras.layers.ReLU()
                                                ])






    def __call__(self, x,training):

        extra_fms=[]
        x=self.extra_conv1(x,training=training)
        extra_fms.append(x)
        x = self.extra_conv2(x,training=training)
        extra_fms.append(x)
        x=self.extra_conv3(x,training=training)
        extra_fms.append(x)

        return extra_fms

class MaxOut(tf.keras.Model):
    def __init__(self,
                 ratio_per_pixel=None,
                 kernel_initializer='glorot_normal'):
        super(MaxOut, self).__init__()

        if ratio_per_pixel is None:
            self.num_predict_per_level = len(cfg.ANCHOR.ANCHOR_RATIOS)
        else:
            self.num_predict_per_level = ratio_per_pixel

        self.neg_conv=tf.keras.layers.Conv2D(filters=self.num_predict_per_level*3,
                                             kernel_size=(3, 3),
                                             strides=1,
                                             padding='same',
                                             kernel_initializer=kernel_initializer
                                             )

        self.pos_conv = tf.keras.layers.Conv2D(filters=self.num_predict_per_level,
                                               kernel_size=(3, 3),
                                               strides=1,
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )


    def call(self,x,training):
        dim_h = tf.shape(x)[1]
        dim_w = tf.shape(x)[2]
        neg_pre=self.neg_conv(x)
        neg_pre_0 = tf.reduce_max(neg_pre[:, :, :, 0:3], axis=3, keepdims=True)
        neg_pre_top3 = tf.reshape(neg_pre_0, ([-1, dim_h, dim_w, self.num_predict_per_level, 1]))

        pos_pre = self.pos_conv(x)
        pos_pre = tf.reshape(pos_pre, ([-1, dim_h, dim_w, self.num_predict_per_level, 1]))

        cls_pre = tf.concat([neg_pre_top3, pos_pre], axis=4)


        return cls_pre

class SSDHead(tf.keras.Model):
    def __init__(self,
                 ratio_per_pixel=None,
                 fm_levels=6,
                 kernel_initializer='glorot_normal'
                 ):
        super(SSDHead, self).__init__()
        if ratio_per_pixel is None:
            self.num_predict_per_level = len(cfg.ANCHOR.ANCHOR_RATIOS)
        else:
            self.num_predict_per_level = ratio_per_pixel

        self.conv_reg = [tf.keras.layers.Conv2D(filters=self.num_predict_per_level * 4,
                                                kernel_size=(3, 3),
                                                padding='same',
                                                kernel_initializer=kernel_initializer
                                                ) for i in range(fm_levels)]

        if cfg.MODEL.maxout:
            self.conv_cls = [MaxOut(ratio_per_pixel=self.num_predict_per_level)]

            self.conv_cls += [tf.keras.layers.Conv2D(filters=self.num_predict_per_level * cfg.DATA.num_class,
                                                     kernel_size=(3, 3),
                                                     padding='same',
                                                     kernel_initializer=kernel_initializer
                                                     ) for i in range(fm_levels - 1)]
        else:
            self.conv_cls = [
                tf.keras.layers.Conv2D(filters=self.num_predict_per_level * cfg.DATA.num_class,
                                       kernel_size=(3, 3),
                                       padding='same',
                                       kernel_initializer=kernel_initializer
                                       ) for i in range(fm_levels)]

    def call(self,fms,training):
        cla_set = []
        reg_set = []


        for i in range(len(fms)):
            current_feature = fms[i]

            dim_h = tf.shape(current_feature)[1]
            dim_w = tf.shape(current_feature)[2]

            reg_out = self.conv_reg[i](current_feature)

            cla_out = self.conv_cls[i](current_feature)

            reg_out = tf.reshape(reg_out, ([-1, dim_h, dim_w, self.num_predict_per_level, 4]))
            reg_out = tf.reshape(reg_out, ([-1, dim_h * dim_w * self.num_predict_per_level, 4]))

            cla_out = tf.reshape(cla_out, ([-1, dim_h, dim_w, self.num_predict_per_level, cfg.DATA.num_class]))
            cla_out = tf.reshape(cla_out, ([-1, dim_h * dim_w * self.num_predict_per_level, cfg.DATA.num_class]))

            cla_set.append(cla_out)
            reg_set.append(reg_out)

        reg = tf.concat(reg_set, axis=1)
        cla = tf.concat(cla_set, axis=1)
        return reg, cla

class VGG(tf.keras.Model):
    def __init__(self,
                 kernel_initializer='glorot_normal'):
        super(VGG, self).__init__()
        self.base_model = tf.keras.applications.VGG16(include_top=False)

        layers_out = ["block3_conv3", "block4_conv3", "block5_conv3"]

        intermid_outputs = [ self.base_model.get_layer(layer_name).output for layer_name in layers_out]
        self.backbone_features = tf.keras.Model(inputs= self.base_model.input, outputs=intermid_outputs)

        self.base_model.summary()
        self.extra=Extra(kernel_initializer=kernel_initializer)

    def call(self,inputs,training):

        p0, p1, p2 = self.backbone_features(inputs, training=training)

        p3, p4, p5 = self.extra(p2, training=training)

        p0 = l2_normalization(p0, scale=cfg.MODEL.l2_norm[0])
        p1 = l2_normalization(p1, scale=cfg.MODEL.l2_norm[1])
        p2 = l2_normalization(p2, scale=cfg.MODEL.l2_norm[2])

        return [p0,p1,p2,p3,p4,p5]

class DSFD(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(DSFD, self).__init__()

        self.base_model = VGG(kernel_initializer=kernel_initializer)





        if cfg.MODEL.fpn:
            self.fpn=Fpn(kernel_initializer=kernel_initializer)
        if cfg.MODEL.cpm:
            self.cpm_ops=[CPM(kernel_initializer=kernel_initializer)
                          for i in range(len(cfg.MODEL.fpn_dims))]

        if cfg.MODEL.dual_mode:
            self.ssd_head_origin=SSDHead(ratio_per_pixel=1,kernel_initializer=kernel_initializer)

            self.ssd_head_fem = SSDHead(ratio_per_pixel=1,kernel_initializer=kernel_initializer)

    def call(self,images,training):

        x=self.preprocess(images)

        vgg_fms=self.base_model(x,training=training)
        if cfg.MODEL.fpn:
            fpn_fms=self.fpn(vgg_fms,training=training)

        fpn_fms[0] = l2_normalization(fpn_fms[0], scale=cfg.MODEL.l2_norm[0])
        fpn_fms[1] = l2_normalization(fpn_fms[1], scale=cfg.MODEL.l2_norm[1])
        fpn_fms[2] = l2_normalization(fpn_fms[2], scale=cfg.MODEL.l2_norm[2])

        if cfg.MODEL.cpm:
            for i in  range(len(fpn_fms)):
                fpn_fms[i]=self.cpm_ops[i](fpn_fms[i],training=training)

        o_reg,o_cls=self.ssd_head_origin(vgg_fms,training=training)

        fpn_reg,fpn_cls=self.ssd_head_fem(fpn_fms,training=training)


        return o_reg,o_cls,fpn_reg,fpn_cls



    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
    def inference(self, images, training=False):

        x = self.preprocess(images)

        vgg_fms = self.base_model(x, training=training)
        if cfg.MODEL.fpn:
            fpn_fms = self.fpn(vgg_fms, training=training)

        fpn_fms[0] = l2_normalization(fpn_fms[0], scale=cfg.MODEL.l2_norm[0])
        fpn_fms[1] = l2_normalization(fpn_fms[1], scale=cfg.MODEL.l2_norm[1])
        fpn_fms[2] = l2_normalization(fpn_fms[2], scale=cfg.MODEL.l2_norm[2])

        if cfg.MODEL.cpm:
            for i in  range(len(fpn_fms)):
                fpn_fms[i]=self.cpm_ops[i](fpn_fms[i],training=training)

        fpn_reg, fpn_cls = self.ssd_head_fem(fpn_fms, training=training)

        ###get anchor
        ###### adjust the anchors to the image shape, but it trains with a fixed h,w

        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        anchors_ = get_all_anchors_fpn(max_size=[h, w])

        if cfg.MODEL.dual_mode:
            anchors_ = anchors_[0::2]
        else:
            anchors_ = anchors_

        res = self.postprocess(fpn_reg, fpn_cls, anchors_)
        return res


    def preprocess(self,image):


        image = tf.cast(image, tf.float32)

        mean = cfg.DATA.PIXEL_MEAN
        std = np.asarray(cfg.DATA.PIXEL_STD)

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean)  *image_invstd

        return image
    def postprocess(self,box_encodings,cla,anchors):
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

            # it has shape [batch_size, num_anchors, 4]

            scores = tf.nn.softmax(cla, axis=2)[:, :, 1:]  ##ignore the bg
            # it has shape [batch_size, num_anchors,class]
            labels = tf.argmax(scores,axis=2)
            # it has shape [batch_size, num_anchors]

            scores = tf.reduce_max(scores,axis=2)
            # it has shape [batch_size, num_anchors]
            scores = tf.expand_dims(scores, axis=-1)
            # it has shape [batch_size, num_anchors]

            res = tf.concat([boxes, scores], axis=2)


        return res


if __name__=='__main__':

    ##test codes for dsfd models
    import time

    model = DSFD()

    image = np.zeros(shape=(1, 320, 320, 3), dtype=np.float32)

    model.inference(image)

    start = time.time()
    for i in range(100):
        model.inference(image)
    print('xxxyyy', (time.time() - start) / 100.)
