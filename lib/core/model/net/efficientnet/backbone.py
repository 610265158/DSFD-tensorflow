import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope



from lib.core.model.net.efficientnet.builder import build_model_base
from lib.core.model.net.FEM import create_fem_net





def efficient_ssd(image,L2_reg,is_training=True,data_format='NHWC'):


    net,endpoints=build_model_base(image,model_name=cfg.MODEL.net_structure,training=is_training)

    for k, v in endpoints.items():
        print('mobile backbone output:', k, v)


    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training)):

        efficient_fms = [endpoints['reduction_2/expansion_output'],
                         endpoints['reduction_3/expansion_output'],
                         endpoints['reduction_4/expansion_output'],
                         endpoints['global_pool']]


        net = slim.conv2d(efficient_fms[-1], 512, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_1_1')
        net = slim.conv2d(net, 512, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_1_2')
        efficient_fms.append(net)
        net = slim.conv2d(net, 128, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_2_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_2_2')
        efficient_fms.append(net)
        print('extra resnet50 backbone output:', efficient_fms)

        if cfg.MODEL.fpn:
            enhanced_fms = create_fem_net(efficient_fms, L2_reg, is_training)
        else:
            enhanced_fms = None
    return efficient_fms,enhanced_fms
