import sys
sys.path.append('.')


import tensorflow as tf
import numpy as np

from lib.core.anchor.box_utils import batch_decode
from lib.core.anchor.nms import batch_non_max_suppression
from lib.core.anchor.tf_anchors import get_all_anchors_fpn

from lib.core.model.shufflenet.shufflenet import Shufflenet

from train_config import config as cfg

def l2_normalization(x, scale):
    x = scale*tf.nn.l2_normalize(x, axis=-1)
    return x


class CPM(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(CPM, self).__init__()

        dim = cfg.MODEL.fem_dims

        self.conv_1_1=tf.keras.layers.Conv2D(filters=dim//2,
                                             kernel_size=(3,3),
                                             dilation_rate=1,
                                             padding='same',
                                             kernel_initializer=kernel_initializer
                                             )


        self.conv_2_1 = tf.keras.layers.Conv2D(filters=dim // 2,
                                               kernel_size=(3, 3),
                                               dilation_rate=2,
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )
        self.conv_2_2 = tf.keras.layers.Conv2D(filters=dim // 4,
                                               kernel_size=(3, 3),
                                               dilation_rate=1,
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )


        self.conv_3_1 = tf.keras.layers.Conv2D(filters=dim // 4,
                                               kernel_size=(3, 3),
                                               dilation_rate=2,
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )
        self.conv_3_2 = tf.keras.layers.Conv2D(filters=dim // 4,
                                               kernel_size=(3, 3),
                                               dilation_rate=1,
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )



    def call(self, x,training):

        cpm1=tf.nn.relu(self.conv_1_1(x))

        cpm_2_1=tf.nn.relu(self.conv_2_1(x))
        cpm_2_2=tf.nn.relu(self.conv_2_2(cpm_2_1))

        cpm_3_1 = tf.nn.relu(self.conv_3_1(cpm_2_1))
        cpm_3_2 = tf.nn.relu(self.conv_3_2(cpm_3_1))

        return tf.concat([cpm1,cpm_2_2,cpm_3_2],axis=3)

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
            self.conv_cls = [tf.keras.layers.Conv2D(filters=self.num_predict_per_level * cfg.DATA.num_class,
                                                    kernel_size=(3, 3),
                                                    padding='same',
                                                    kernel_initializer=kernel_initializer
                                                    ) for i in range(fm_levels-1)]
            self.conv_cls.insert(0,MaxOut(ratio_per_pixel=self.num_predict_per_level))
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


class DSFD(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(DSFD, self).__init__()

        self.base_model = Shufflenet(model_size='0.5',
                                     kernel_initializer=kernel_initializer)





        if cfg.MODEL.dual_mode:
            self.ssd_head_origin=SSDHead(ratio_per_pixel=1,
                                         fm_levels=5,
                                         kernel_initializer=kernel_initializer)

            self.ssd_head_fem = SSDHead(ratio_per_pixel=1,
                                        fm_levels=5,
                                        kernel_initializer=kernel_initializer)

    def call(self,images,training):

        x=self.preprocess(images)

        net,end_points=self.base_model(x,training=training)


        fms=[#end_points['block0'],
             end_points['block1'],
             end_points['block2'],
             end_points['block3'],
             end_points['block4'],
             end_points['block5']]

        # shapes = [y.shape for y in fms]
        # print(shapes)


        o_reg,o_cls=self.ssd_head_origin(fms,training=training)

        fpn_reg,fpn_cls=self.ssd_head_fem(fms,training=training)

        return o_reg,o_cls,fpn_reg,fpn_cls


    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
    def inference(self,images,training=False):

        x = self.preprocess(images)

        net, end_points = self.base_model(x, training=training)

        fms = [#end_points['block0'],
               end_points['block1'],
               end_points['block2'],
               end_points['block3'],
               end_points['block4'],
               end_points['block5']]


        fpn_reg, fpn_cls = self.ssd_head_fem(fms, training=training)

        ###get anchor
        ###### adjust the anchors to the image shape, but it trains with a fixed h,w

        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        anchors_ = get_all_anchors_fpn(max_size=[h, w])


        if cfg.MODEL.dual_mode:
            anchors_ = anchors_[0::2]
        else:
            anchors_ = anchors_

        res=self.postprocess(fpn_reg, fpn_cls, anchors_)
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

        return {'boxes': boxes, 'scores': scores,'labels':labels, 'num_boxes': num_detections}


if __name__=='__main__':



    ##teset codes for dsfd models
    import time
    model=DSFD()

    image = np.zeros(shape=(1, 240, 320, 3), dtype=np.float32)

    model.inference(image)

    start = time.time()
    for i in range(100):
        model.inference(image)
    print('xxxyyy', (time.time() - start) / 100.)
