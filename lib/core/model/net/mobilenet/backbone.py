import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg

from lib.core.model.net.mobilenet.mobilenet_v1 import mobilenet_v1_050,mobilenet_v1_arg_scope
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope


from lib.core.model.net.FEM import create_fem_net

def mobilenet_ssd(image,L2_reg,is_training=True,data_format='NHWC'):


    assert 'MobilenetV1' in cfg.MODEL.net_structure
    if cfg.TRAIN.lock_basenet_bn:
        arg_scope = mobilenet_v1_arg_scope(weight_decay=L2_reg, is_training=False)
    else:
        arg_scope = mobilenet_v1_arg_scope(weight_decay=L2_reg, is_training=is_training)


    with tf.contrib.slim.arg_scope(arg_scope):
        _,endpoint = mobilenet_v1_050(image,is_training=is_training,num_classes=None,global_pool=False)

    for k,v in endpoint.items():
        print('mobile backbone output:',k,v)



    mobilebet_fms=[endpoint['Conv2d_3_pointwise'],endpoint['Conv2d_5_pointwise'],endpoint['Conv2d_11_pointwise'],endpoint['Conv2d_13_pointwise']]

    print('mobile backbone output:',mobilebet_fms)
    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training)):

        # net = block(resnet_fms[-1], num_units=2, out_channels=512, scope='extra_Stage1')
        # resnet_fms.append(net)
        # net = block(net, num_units=2, out_channels=512, scope='extra_Stage2')
        # resnet_fms.append(net)
        net = slim.conv2d(mobilebet_fms[-1], 512, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_1_1')
        net = slim.conv2d(net, 512, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_1_2')
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
