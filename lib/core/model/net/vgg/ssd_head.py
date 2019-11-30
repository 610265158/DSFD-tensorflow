#-*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope
from train_config import config as cfg

num_predict_per_level=len(cfg.ANCHOR.ANCHOR_RATIOS)


class SSDHead():

    def max_out_cla(self,fm,ratios_per_pixel,cla_num=2,scope='nul'):
        dim_h = tf.shape(fm)[1]
        dim_w = tf.shape(fm)[2]
        with tf.variable_scope(scope):
            with tf.variable_scope('negative'):
                neg_pre = slim.conv2d(fm, num_predict_per_level*3, [3, 3], stride=1, activation_fn=None,
                                        normalizer_fn=None, scope='out_cla0')
                if ratios_per_pixel==3:
                    neg_pre_0 = tf.reduce_max(neg_pre[:, :, :, 0:3], axis=3,keepdims=True)
                    neg_pre_1 = tf.reduce_max(neg_pre[:, :, :, 3:6], axis=3,keepdims=True)
                    neg_pre_2 = tf.reduce_max(neg_pre[:, :, :, 6:9], axis=3,keepdims=True)

                    neg_pre_top3 =tf.concat([neg_pre_0,neg_pre_1,neg_pre_2],axis=3)
                    neg_pre_top3 = tf.reshape(neg_pre_top3, ([-1, dim_h, dim_w, ratios_per_pixel, 1]))

                elif ratios_per_pixel==2:
                    neg_pre_0 = tf.reduce_max(neg_pre[:, :, :, 0:3], axis=3,keepdims=True)
                    neg_pre_1 = tf.reduce_max(neg_pre[:, :, :, 3:6], axis=3,keepdims=True)

                    neg_pre_top3 =tf.concat([neg_pre_0,neg_pre_1],axis=3)
                    neg_pre_top3 = tf.reshape(neg_pre_top3, ([-1, dim_h, dim_w, ratios_per_pixel, 1]))
                else:
                    neg_pre_0 = tf.reduce_max(neg_pre[:, :, :, 0:3], axis=3, keepdims=True)

                    neg_pre_top3 = tf.reshape(neg_pre_0, ([-1, dim_h, dim_w, ratios_per_pixel, 1]))
            with tf.variable_scope('positive'):
                pos_pre = slim.conv2d(fm, ratios_per_pixel*(cla_num-1), [3, 3], stride=1, activation_fn=None,
                                                normalizer_fn=None, scope='out_cla1')
                pos_pre = tf.reshape(pos_pre, ([-1, dim_h, dim_w, ratios_per_pixel, (cla_num-1)]))

            cla_pre=tf.concat([neg_pre_top3,pos_pre],axis=4)


        return cla_pre


    def __call__(self,fms,L2_reg,training=True,ratios_per_pixel=num_predict_per_level):

        cla_set=[]
        reg_set=[]
        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('ssdout'):

                for i in range(len(fms)):
                    current_feature = fms[i]

                    dim_h=tf.shape(current_feature)[1]
                    dim_w = tf.shape(current_feature)[2]
                    #feature_halo=halo(current_feature,'fm%d'%i)
                    feature_halo=current_feature
                    reg_out = slim.conv2d(feature_halo, ratios_per_pixel*4, [3, 3], stride=1, activation_fn=None,normalizer_fn=None, scope='out_reg%d'%i)

                    if cfg.MODEL.maxout and i==0:
                        cla_out = self.max_out_cla(feature_halo,ratios_per_pixel,cla_num=cfg.DATA.num_class,scope='maxout%d'%i)
                    else:
                        cla_out = slim.conv2d(feature_halo, ratios_per_pixel*cfg.DATA.num_class, [3, 3], stride=1, activation_fn=None,normalizer_fn=None, scope='out_cla%d'%i)

                    reg_out = tf.reshape(reg_out, ([-1, dim_h, dim_w, ratios_per_pixel, 4]))
                    reg_out = tf.reshape(reg_out, ([-1, dim_h * dim_w * ratios_per_pixel, 4]))

                    cla_out = tf.reshape(cla_out, ([-1, dim_h, dim_w, ratios_per_pixel, cfg.DATA.num_class]))
                    cla_out = tf.reshape(cla_out, ([-1, dim_h * dim_w* ratios_per_pixel,cfg.DATA.num_class]))




                    cla_set.append(cla_out)
                    reg_set.append(reg_out)



                reg = tf.concat(reg_set, axis=1)
                cla = tf.concat(cla_set, axis=1)
        return reg,cla





