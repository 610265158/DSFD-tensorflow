import sys
sys.path.append('.')


import tensorflow as tf
import numpy as np

from lib.core.anchor.box_utils import batch_decode
from lib.core.anchor.nms import batch_non_max_suppression
from lib.core.anchor.tf_anchors import get_all_anchors_fpn

from lib.core.model.light.lightnet import Lightnet

from train_config import config as cfg

def l2_normalization(x, scale):
    x = scale*tf.nn.l2_normalize(x, axis=-1)
    return x


def batch_norm():
    return tf.keras.layers.BatchNormalization(fused=True,momentum=0.997,epsilon=1e-5)

class CPM(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(CPM, self).__init__()

        dim = cfg.MODEL.cpm_dims

        self.conv1=tf.keras.Sequential([tf.keras.layers.Conv2D(filters=dim//2,
                                                               kernel_size=(1,1),
                                                               padding='same',
                                                               kernel_initializer=kernel_initializer,
                                                               use_bias=False),
                                           batch_norm()])



        self.conv2 = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(filters=dim//4,
                                                                          kernel_size=(3,3),
                                                                          padding='same',
                                                                          kernel_initializer=kernel_initializer,
                                                                          use_bias=False),
                                             batch_norm()])


        self.conv3 = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(filters=dim//4,
                                                                          kernel_size=(3,3),
                                                                          padding='same',
                                                                          kernel_initializer=kernel_initializer,
                                                                          use_bias=False),
                                             batch_norm()])


    def call(self, x,training):

        cpm1=self.conv1(x,training=training)

        cpm2=self.conv2(x,training=training)

        cpm3 =self.conv3(cpm2,training=training)

        return tf.nn.relu(tf.concat([cpm1,cpm2,cpm3],axis=3))

class Fpn(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(Fpn, self).__init__()

        dims_list = cfg.MODEL.fpn_dims


        self.conv_1_0 = tf.keras.layers.Conv2D(filters=dims_list[0],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer,

                                               )
        self.conv_1_1 = tf.keras.layers.Conv2D(filters=dims_list[0],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer,
                                               )

        self.upsample=tf.keras.layers.UpSampling2D()


    def __call__(self, fms,training):


        of1,of2,of3=fms



        upsample_2 = self.conv_1_0(of2)

        lateral_1 = self.conv_1_1(of1)
        fpn1 = self.upsample_add(upsample_2, lateral_1)


        return [fpn1,upsample_2,of3]



    def upsample_product(self,x,y):

        x_upsample=self.upsample(x)
        return x_upsample*y

    def upsample_add(self,x,y):

        x_upsample=self.upsample(x)
        return x_upsample+y

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
                                             kernel_initializer=kernel_initializer,
                                             use_bias=False
                                             )

        self.pos_conv = tf.keras.layers.Conv2D(filters=self.num_predict_per_level,
                                               kernel_size=(3, 3),
                                               strides=1,
                                               padding='same',
                                               kernel_initializer=kernel_initializer,
                                               use_bias=False
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
class Extra(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(Extra, self).__init__()

        self.extra_conv1=tf.keras.Sequential([tf.keras.layers.Conv2D(filters=128,
                                                                     kernel_size=(1, 1),
                                                                     padding='same',
                                                                     kernel_initializer=kernel_initializer,
                                                                     use_bias=False),
                                                batch_norm(),
                                                tf.keras.layers.ReLU(),

                                                tf.keras.layers.SeparableConv2D(filters=256,
                                                                                kernel_size=(3, 3),
                                                                                strides=2,
                                                                                padding='same',
                                                                                kernel_initializer=kernel_initializer,
                                                                                use_bias=False),
                                                batch_norm(),
                                                tf.keras.layers.ReLU()
                                                ])


    def __call__(self, x,training):

        x1=self.extra_conv1(x,training=training)

        return x1

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
                                                kernel_size=(1, 1),
                                                padding='same',
                                                kernel_initializer=kernel_initializer
                                                ) for i in range(fm_levels)]


        self.conv_cls = [tf.keras.layers.Conv2D(filters=self.num_predict_per_level * cfg.DATA.num_class,
                                                kernel_size=(1, 1),
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


class DSFD(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(DSFD, self).__init__()


        
        model_size=cfg.MODEL.net_structure.split('_',1)[-1]
        self.base_model = Lightnet(model_size=model_size,
                                     kernel_initializer=kernel_initializer)
        self.extra=Extra(kernel_initializer=kernel_initializer)


        if cfg.MODEL.fpn:
            self.fpn=Fpn()

        if cfg.MODEL.cpm:
            self.cpm_ops = [CPM(kernel_initializer=kernel_initializer)
                            for i in range(len(cfg.MODEL.fpn_dims))]

        if cfg.MODEL.dual_mode:
            self.ssd_head_origin=SSDHead(ratio_per_pixel=1,
                                         fm_levels=3,
                                         kernel_initializer=kernel_initializer)

            self.ssd_head_fem = SSDHead(ratio_per_pixel=1,
                                        fm_levels=3,
                                        kernel_initializer=kernel_initializer)

        else:
            self.ssd_head_fem = SSDHead(ratio_per_pixel=2,
                                        fm_levels=3,
                                        kernel_initializer=kernel_initializer)


    def call(self,images,training):

        x=self.preprocess(images)

        of1,of2=self.base_model(x,training=training)

        of3=self.extra(of2, training=training)

        fms=[of1,of2,of3]

        if cfg.MODEL.dual_mode:
            o_reg, o_cls=self.ssd_head_origin(fms,training=training)
        else:
            o_reg=None
            o_cls=None

        if cfg.MODEL.fpn:
            fpn_fms = self.fpn(fms, training=False)
        else:
            fpn_fms = fms


        if cfg.MODEL.cpm:
            for i in range(len(fpn_fms)):
                fpn_fms[i] = self.cpm_ops[i](fpn_fms[i], training=training)

        fpn_reg,fpn_cls=self.ssd_head_fem(fpn_fms,training=training)

        return o_reg,o_cls,fpn_reg,fpn_cls


    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32),
                                  tf.TensorSpec(None, tf.float32),
                                  tf.TensorSpec(None, tf.float32)
                                  ])
    def inference(self, images,
                        score_threshold=cfg.TEST.score_thres, \
                        iou_threshold=cfg.TEST.iou_thres):

        x = self.preprocess(images)

        of1, of2 = self.base_model(x, training=False)

        of3 = self.extra(of2, training=False)

        fms = [of1, of2, of3]

        for i in range(len(fms)):
            print(fms[i].shape)

        if cfg.MODEL.fpn:
            fpn_fms = self.fpn(fms, training = False)
        else:
            fpn_fms=fms


        if cfg.MODEL.cpm:
            for i in range(len(fpn_fms)):
                print(fpn_fms[i].shape)
                fpn_fms[i] = self.cpm_ops[i](fpn_fms[i], training=False)


        fpn_reg, fpn_cls = self.ssd_head_fem(fpn_fms, training=False)



        ###get anchor
        ###### adjust the anchors to the image shape, but it trains with a fixed h,w

        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        anchors_ = get_all_anchors_fpn(max_size=[h, w])


        if cfg.MODEL.dual_mode:
            anchors_ = anchors_[0::2]
        else:
            anchors_ = anchors_

        res=self.postprocess(fpn_reg, fpn_cls, anchors_,score_threshold,iou_threshold)
        return res


    def preprocess(self,image):


        image = tf.cast(image, tf.float32)

        mean = cfg.DATA.PIXEL_MEAN
        std = np.asarray(cfg.DATA.PIXEL_STD)

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean)  *image_invstd

        return image
    def postprocess(self,box_encodings,cla,anchors, \
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

            # it has shape [batch_size, num_anchors, 4]

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


        return {'boxes': boxes, 'scores': scores,'labels':labels, 'num_boxes': num_detections}




if __name__=='__main__':



    ####test codes for dsfd models
    import time
    model=DSFD()

    image = np.zeros(shape=(1, 320, 320, 3), dtype=np.float32)

    model.inference(image,0.5,0.45)

    #tf.saved_model.save(model, './model/xx')
    start = time.time()
    for i in range(100):
        model.inference(image,0.5,0.45)
    print('xxxyyy', (time.time() - start) / 100.)
