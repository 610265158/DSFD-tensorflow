#-*-coding:utf-8-*-


import tensorflow as tf


def GroupNorm(x, group=16, gamma_initializer=tf.constant_initializer(1.),scope='GN'):
    """
    https://arxiv.org/abs/1803.08494
    """

    ##to  nchw
    x=tf.transpose(x, [0, 3, 1, 2])
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    chan = shape[1]

    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    with tf.variable_scope(scope):
        beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)

        gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
        gamma = tf.reshape(gamma, new_shape)

        out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')

    after_norm=tf.reshape(out, orig_shape, name='output')

    ##to nhwc
    after_norm=tf.transpose(after_norm, [0, 2, 3, 1])

    return after_norm
