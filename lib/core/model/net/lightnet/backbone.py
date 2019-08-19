#-*-coding:utf-8-*-


import tensorflow as tf

import tensorflow.contrib.slim as slim

from lib.core.model.net import shufflenet_arg_scope

from lib.core.model.net import simple_nn

def create_fpn_net(blocks, L2_reg,is_training, trainable=True,data_format='NHWC'):

    global_fms = []

    last_fm = None
    initializer = tf.contrib.layers.xavier_initializer()
    for i, block in enumerate(reversed(blocks)):
        with slim.arg_scope(shufflenet_arg_scope(weight_decay=L2_reg,is_training=is_training)):
            lateral = slim.conv2d(block, 256, [1, 1],
                trainable=trainable, weights_initializer=initializer,
                padding='SAME', activation_fn=None,
                scope='lateral/res{}'.format(5-i))

            if last_fm is not None and i >=3:

                upsample = tf.keras.layers.UpSampling2D(data_format='channels_last' if data_format=='NHWC' else 'channels_first')(last_fm)
                # upsample = slim.conv2d(upsample, 10, [1, 1],
                #     trainable=trainable, weights_initializer=initializer,
                #     padding='SAME', activation_fn=None,
                #     scope='merge/res{}'.format(5-i),data_format=data_format)

                last_fm = upsample + lateral
            else:
                last_fm = lateral

        global_fms.append(tf.nn.relu(last_fm,name='fpn_relu/res{}'.format(5-i)))

    global_fms.reverse()

    return global_fms



def fpn(image,L2_reg,is_training=True,data_format='NHWC'):


    net_fms = simple_nn(image, L2_reg,is_training)

    print('simplenet backbone output:',net_fms)

    # with tf.variable_scope('CPN'):
    fpn_fms = create_fpn_net(net_fms, L2_reg,is_training,data_format=data_format)

    return fpn_fms



