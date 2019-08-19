#-*-coding:utf-8-*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
from train_config import config as cfg

from lib.core.model.net.GN import GroupNorm

def shufflenet_arg_scope(weight_decay=0.00001,
                     batch_norm_decay=0.99,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     use_batch_norm=True,
                     is_training=True,
                     trainable=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
                     ):
  """Defines the default ResNet arg scope.
  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': batch_norm_updates_collections,
      'fused': True,  # Use fused batch norm if possible.
      'trainable':True,
      'is_training':is_training
  }

  with slim.arg_scope(
      [slim.conv2d,slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      normalizer_fn=slim.batch_norm if 'BN' in cfg.TRAIN.norm else GroupNorm,
      normalizer_params=batch_norm_params if 'BN' in cfg.TRAIN.norm else None,
      biases_initializer=None):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc

def residual_simple(x):

    depth_in=x.shape[3]
    shortcut=slim.conv2d(x, depth_in, [1, 1], stride=1,activation_fn=None,scope='shortcut')

    residual = slim.conv2d(x, depth_in, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='init_conv2_1')
    residual = slim.conv2d(residual, depth_in, [3, 3], stride=1, activation_fn=None, scope='init_conv2_2')

    residual_unit=shortcut+residual
    act = slim.batch_norm(residual_unit, activation_fn=tf.nn.relu, scope='act')
    return  act

def residual_dense(x):
    depth_in = x.shape[3]
    shortcut = slim.conv2d(x, depth_in//2, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='shortcut')

    residual = slim.conv2d(x, depth_in//2, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='init_conv2_1')
    residual = slim.conv2d(residual, depth_in//2, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='init_conv2_2')

    x = tf.concat([shortcut, residual], axis=3)
    return x

def halo_resisual(x,out_channels,scope):

    with tf.variable_scope(scope):
        with tf.variable_scope('first_branch'):
            x1 = slim.conv2d(x, out_channels//2, [3, 3], stride=2, activation_fn=None, scope='_conv_1_1')
        with tf.variable_scope('second_branch'):
            x2 = slim.conv2d(x, out_channels // 2, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='_conv_2_1')
            x2 = slim.conv2d(x2, out_channels//2, [3, 3], stride=2, activation_fn=None, scope='_conv_2_2')
    residual_unit = x1 + x2
    x = slim.batch_norm(residual_unit, activation_fn=tf.nn.relu, scope='act')
    return x

def halo(x,out_channels,scope):

    with tf.variable_scope(scope):
        with tf.variable_scope('first_branch'):
            x1 = slim.conv2d(x, out_channels//2, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='_conv_1_1')
        with tf.variable_scope('second_branch'):
            x2 = slim.conv2d(x, out_channels // 2, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='_conv_2_1')
            x2 = slim.conv2d(x2, out_channels//2, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='_conv_2_2')
    x = tf.concat([x1, x2], axis=3)
    return x


def block(x,num_units,out_channels,scope):
    with tf.variable_scope(scope):
        x=halo(x,out_channels,scope)

        for i in range(num_units-1):
            with tf.variable_scope('residul_%d'%i):
                x=residual_dense(x)
    return x


def magic_nn(inputs,L2_reg,training=True):

    fms=[]

    arg_scope = shufflenet_arg_scope(weight_decay=L2_reg,is_training=training,)
    with slim.arg_scope(arg_scope):

        with tf.variable_scope('simpleface'):
            net = slim.conv2d(inputs, 32, [7, 7],stride=2, activation_fn=tf.nn.relu, scope='init_conv0')

            net = block(net, num_units=2, out_channels=64, scope='Stage1')
            print('1 conv shape', net.shape)
            fms.append(net)

            net = block(net, num_units=2, out_channels=128, scope='Stage2')
            print('2 conv shape', net.shape)
            fms.append(net)

            net = block(net, num_units=2, out_channels=256, scope='Stage3')
            print('3 conv shape', net.shape)
            fms.append(net)

            net = block(net, num_units=2, out_channels=512, scope='Stage4')
            print('4 conv shape', net.shape)
            fms.append(net)

            net = block(net, num_units=2, out_channels=1024, scope='Stage5')
            print('5 conv shape', net.shape)
            fms.append(net)

            net = block(net, num_units=2, out_channels=1024, scope='Stage6')
            print('6 conv shape', net.shape)
            fms.append(net)
            ##use three layes feature in different scale 28x28 14x14 7x7



    return fms


