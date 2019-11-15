import sys
sys.path.append('.')


import tensorflow as tf
import numpy as np



from lib.core.anchor.box_utils import batch_decode
from lib.core.anchor.tf_anchors import get_all_anchors_fpn

from lib.core.model.mobilenet.mb import MobileNet

from train_config import config as cfg

def l2_normalization(x, scale):
    x = scale*tf.nn.l2_normalize(x, axis=-1)
    return x


def batch_norm():
    return tf.keras.layers.BatchNormalization(fused=True,momentum=0.997,epsilon=1e-5)

class CPM(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal',dim=256, endhance=True):
        super(CPM, self).__init__()


        # self.conv1=tf.keras.Sequential([tf.keras.layers.Conv2D(filters=dim//2,
        #                                                        kernel_size=(1,1),
        #                                                        padding='same',
        #                                                        kernel_initializer=kernel_initializer,
        #                                                        use_bias=False),
        #                                    batch_norm()])



        self.conv2 = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(filters=dim//4,
                                                                          kernel_size=(3,3),
                                                                          dilation_rate=3,
                                                                          padding='same',
                                                                          kernel_initializer=kernel_initializer,
                                                                          use_bias=False),
                                             batch_norm()])


        self.conv3 = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(filters=dim//4,
                                                                          kernel_size=(3,3),
                                                                          dilation_rate=3,
                                                                          padding='same',
                                                                          kernel_initializer=kernel_initializer,
                                                                          use_bias=False),
                                             batch_norm()])


    def call(self, x,training):

        #cpm1=self.conv1(x,training=training)

        cpm2=self.conv2(x,training=training)

        cpm3 =self.conv3(cpm2,training=training)

        return tf.nn.relu(tf.concat([x,cpm2,cpm3],axis=3))

class Fpn(tf.keras.Model):
    def __init__(self,kernel_initializer='glorot_normal'):
        super(Fpn, self).__init__()

        dims_list = cfg.MODEL.fpn_dims

        self.conv_2_0 = tf.keras.layers.Conv2D(filters=dims_list[1],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )
        self.conv_2_1 = tf.keras.layers.Conv2D(filters=dims_list[1],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )

        self.conv_1_0 = tf.keras.layers.Conv2D(filters=dims_list[0],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )
        self.conv_1_1 = tf.keras.layers.Conv2D(filters=dims_list[0],
                                               kernel_size=(1, 1),
                                               padding='same',
                                               kernel_initializer=kernel_initializer
                                               )

        self.upsample=tf.keras.layers.UpSampling2D()
    def __call__(self, fms,training):


        of1,of2,of3=fms

        of3_upsample = self.conv_2_0(of3)

        lateral = self.conv_2_1(of2)
        fpn2 = self.upsample_product(of3_upsample, lateral)

        upsample = self.conv_1_0(fpn2)

        lateral = self.conv_1_1(of1)
        fpn1 = self.upsample_product(upsample, lateral)

        return [fpn1,fpn2,of3_upsample]



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
                                             kernel_size=(1, 1),
                                             strides=1,
                                             padding='same',
                                             kernel_initializer=kernel_initializer,
                                             use_bias=False
                                             )

        self.pos_conv = tf.keras.layers.Conv2D(filters=self.num_predict_per_level,
                                               kernel_size=(1, 1),
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

                                                tf.keras.layers.SeparableConv2D(filters=192,
                                                                                kernel_size=(3, 3),
                                                                                strides=2,
                                                                                padding='same',
                                                                                kernel_initializer=kernel_initializer,
                                                                                use_bias=False),
                                                batch_norm(),
                                                tf.keras.layers.ReLU()
                                                ])

        # self.extra_conv2 = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=128,
        #                                                                kernel_size=(1, 1),
        #                                                                padding='same',
        #                                                                kernel_initializer=kernel_initializer,
        #                                                                use_bias=False),
        #                                         batch_norm(),
        #                                         tf.keras.layers.ReLU(),
        #
        #                                         tf.keras.layers.SeparableConv2D(filters=192,
        #                                                                         kernel_size=(3, 3),
        #                                                                         strides=2,
        #                                                                         padding='same',
        #                                                                         kernel_initializer=kernel_initializer,
        #                                                                         use_bias=False),
        #                                         batch_norm(),
        #                                         tf.keras.layers.ReLU()
        #                                         ])


    def __call__(self, x,training):

        x1=self.extra_conv1(x,training=training)

        #x2 = self.extra_conv2(x1, training=training)
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

        if cfg.MODEL.focal_loss:
            self.num_class = (cfg.DATA.num_class - 1)
        else:
            self.num_class = (cfg.DATA.num_class)


        self.conv_reg = [tf.keras.layers.Conv2D(filters=self.num_predict_per_level * 4,
                                                kernel_size=(1, 1),
                                                padding='same',
                                                kernel_initializer=kernel_initializer
                                                ) for i in range(fm_levels)]



        self.conv_cls = [
            tf.keras.layers.Conv2D(filters=self.num_predict_per_level * (self.num_class),
                                   kernel_size=(1, 1),
                                   padding='same',
                                   kernel_initializer=kernel_initializer
                                   ) for i in range(fm_levels)]



    def call(self,fms,training):
        cls_set = []
        reg_set = []


        for i in range(len(fms)):
            current_feature = fms[i]

            dim_h = tf.shape(current_feature)[1]
            dim_w = tf.shape(current_feature)[2]

            reg_out = self.conv_reg[i](current_feature)

            cls_out = self.conv_cls[i](current_feature)

            reg_out = tf.reshape(reg_out, ([-1, dim_h, dim_w, self.num_predict_per_level, 4]))
            reg_out = tf.reshape(reg_out, ([-1, dim_h * dim_w * self.num_predict_per_level, 4]))


            cls_out = tf.reshape(cls_out, ([-1, dim_h, dim_w, self.num_predict_per_level, self.num_class]))
            cls_out = tf.reshape(cls_out, ([-1, dim_h * dim_w * self.num_predict_per_level, self.num_class]))

            cls_set.append(cls_out)
            reg_set.append(reg_out)

        reg = tf.concat(reg_set, axis=1)
        cls = tf.concat(cls_set, axis=1)
        return reg, cls


class DSFD(tf.keras.Model):
    def __init__(self, kernel_initializer='glorot_normal'):
        super(DSFD, self).__init__()

        model_size = float(cfg.MODEL.net_structure.split('_', 1)[-1])
        self.base_model = MobileNet(model_size=model_size)

        if cfg.MODEL.fpn:
            self.fpn = Fpn()

        if cfg.MODEL.cpm:
            self.cpm_ops = [CPM(kernel_initializer=kernel_initializer, dim=cfg.MODEL.cpm_dims[i])
                            for i in range(len(cfg.MODEL.fpn_dims))]

        if cfg.MODEL.dual_mode:
            self.ssd_head_origin = SSDHead(ratio_per_pixel=2,
                                           fm_levels=3,
                                           kernel_initializer=kernel_initializer)

            self.ssd_head_fem = SSDHead(ratio_per_pixel=2,
                                        fm_levels=3,
                                        kernel_initializer=kernel_initializer)

        else:
            self.ssd_head_fem = SSDHead(ratio_per_pixel=2,
                                        fm_levels=3,
                                        kernel_initializer=kernel_initializer)

    def call(self, images, training):

        x = self.preprocess(images)

        of1, of2, of3 = self.base_model(x, training=training)

        fms = [of1, of2, of3]

        # for i in range(len(fms)):
        #     print(fms[i].shape)

        if cfg.MODEL.dual_mode:
            o_reg, o_cls = self.ssd_head_origin(fms, training=training)
        else:
            o_reg = None
            o_cls = None

        if cfg.MODEL.fpn:
            fpn_fms = self.fpn(fms, training=training)
        else:
            fpn_fms = fms

        if cfg.MODEL.cpm:
            for i in range(len(fpn_fms)):
                fpn_fms[i] = self.cpm_ops[i](fpn_fms[i], training=training)

        fpn_reg, fpn_cls = self.ssd_head_fem(fpn_fms, training=training)

        return o_reg, o_cls, fpn_reg, fpn_cls

    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
    def inference(self, images):

        x = self.preprocess(images)

        of1, of2, of3 = self.base_model(x, training=False)

        fms = [of1, of2, of3]

        if cfg.MODEL.fpn:
            fpn_fms = self.fpn(fms, training=False)
        else:
            fpn_fms = fms

        if cfg.MODEL.cpm:
            for i in range(len(fpn_fms)):
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

        res = self.postprocess(fpn_reg, fpn_cls, anchors_)
        return res

    def preprocess(self, image):

        mean = cfg.DATA.PIXEL_MEAN
        std = cfg.DATA.PIXEL_STD
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_invstd

        return image

    def postprocess(self, box_encodings, cls, anchors):
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
            if cfg.MODEL.focal_loss:
                scores = tf.nn.sigmoid(cls)  ##ignore the bg
            else:
                scores = tf.nn.softmax(cls, axis=2)[:, :, 1:]  ##ignore the bg

                # it has shape [batch_size, num_anchors,class]
                labels = tf.argmax(scores, axis=2)
                # it has shape [batch_size, num_anchors]

                scores = tf.reduce_max(scores, axis=2)
                # it has shape [batch_size, num_anchors]
                scores = tf.expand_dims(scores, axis=-1)
                # it has shape [batch_size, num_anchors]

            res = tf.concat([boxes, scores], axis=2)

        return res


if __name__ == '__main__':

    ####test codes for dsfd models
    import time

    model = DSFD()

    image = np.zeros(shape=(1, 320, 320, 3), dtype=np.float32)

    model.inference(image)

    tf.saved_model.save(model, './model/xx')
    start = time.time()
    for i in range(100):
        model.inference(image)
    print('xxxyyy', (time.time() - start) / 100.)
