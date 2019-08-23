import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg

from lib.core.model.net.mobilenet.mobilenet_v2 import mobilenet_v2_050
from lib.core.model.net.mobilenet.mobilenet import training_scope


from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope
from lib.core.model.net.FEM import create_fem_net

def mobilenet_ssd(image,L2_reg,is_training=True):


    assert 'MobilenetV2' in cfg.MODEL.net_structure

    if cfg.TRAIN.lock_basenet_bn:
        arg_scope = training_scope(weight_decay=L2_reg, is_training=False)
    else:
        arg_scope = training_scope(weight_decay=L2_reg, is_training=is_training)

    with tf.contrib.slim.arg_scope(arg_scope):
        _,endpoint = mobilenet_v2_050(image,is_training=is_training,base_only=True,finegrain_classification_mode=False)

    for k,v in endpoint.items():
        print('mobile backbone output:',k,v)

    mobilebet_fms=[endpoint['layer_5/expansion_output'],
                   endpoint['layer_8/expansion_output'],
                   endpoint['layer_15/expansion_output'],
                   endpoint['layer_18/output']]

    print('mobile backbone output:',mobilebet_fms)
    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training)):


        net = slim.conv2d(mobilebet_fms[-1], 128, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_1_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_1_2')
        mobilebet_fms.append(net)
        net = slim.conv2d(net, 128, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_2_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_2_2')
        mobilebet_fms.append(net)
        print('extra backbone output:', mobilebet_fms)
        if cfg.MODEL.fpn:
            enhanced_fms = create_fem_net(mobilebet_fms, L2_reg, is_training)
        else:
            enhanced_fms =None
    return mobilebet_fms,enhanced_fms
