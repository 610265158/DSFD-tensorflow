
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import regularizers, \
    layers
from train_config import config


from lib.core.model.net.GN import GroupNorm


def resnet_arg_scope(bn_is_training,
                     bn_trainable=True,
                     trainable=True,
                     weight_decay=config.TRAIN.weight_decay_factor,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-9,
                     batch_norm_scale=True,
                     bn_method='BN',
                     data_format='NHWC'):
    batch_norm_params = {
        'is_training': bn_is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': bn_trainable,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
        'fused':True
    }
    if 'BN' in config.TRAIN.norm:
        norm_func=slim.batch_norm
        norm_params=batch_norm_params
    elif 'GN' in config.TRAIN.norm or 'GN' in bn_method:
        norm_func =GroupNorm
        norm_params = None
    elif 'None' in config.TRAIN.norm or 'None' in bn_method :
        norm_func = None
        norm_params = None

    with arg_scope(
            [slim.conv2d,slim.separable_conv2d],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=slim.xavier_initializer(),
            trainable=trainable,
            activation_fn=nn_ops.relu,
            normalizer_fn=norm_func,
            normalizer_params=norm_params,
            data_format=data_format,):
        with arg_scope(
                [layers.batch_norm,layers.max_pool2d], data_format=data_format):
            with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:

                return arg_sc
