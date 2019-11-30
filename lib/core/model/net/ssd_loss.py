#-*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops

from train_config import config as cfg

def ssd_loss(reg_predict,cla_predict,reg_label,cla_label,which_loss='focal_loss'):

    with tf.name_scope('losses'):
        # whether anchor is matched
        is_matched = tf.greater(cla_label, 0)
        weights = tf.to_float(is_matched)
        # shape [batch_size, num_anchors]

        with tf.name_scope('classification_loss'):
            if 'focal_loss' in which_loss:
                cls_losses = focal_loss(
                    cla_predict,
                    cla_label
                )
            else:
                cls_losses = ohem_loss(
                    cla_predict,
                    cla_label,
                    weights
                )

        with tf.name_scope('localization_loss'):
            location_losses = localization_loss(
                reg_predict,
                reg_label, weights
            )


    with tf.name_scope('normalization'):
        matches_per_image = tf.reduce_sum(weights, axis=1)  # shape [batch_size]
        num_matches = tf.reduce_sum(matches_per_image)  # shape []
        normalizer = tf.maximum(num_matches, 1.0)


    reg_loss = tf.reduce_sum(location_losses) / normalizer
    cla_loss = tf.reduce_sum(cls_losses)/ normalizer

    return reg_loss,cla_loss





def classification_loss(predictions, targets):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes + 1],
            representing the predicted logits for each class.
        targets: an int tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=predictions
    )
    return cross_entropy


def localization_loss(predictions, targets, weights,sigma=9):
    """A usual L1 smooth loss.

    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, 4],
            representing the (encoded) predicted locations of objects.
        targets: a float tensor with shape [batch_size, num_anchors, 4],
            representing the regression targets.
        weights: a float tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    abs_diff = tf.abs(predictions - targets)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0/sigma)
    return weights * tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5/sigma), axis=2
    )


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    target_tensor=tf.one_hot(target_tensor,depth=cfg.DATA.num_class)

    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    return tf.reduce_sum(per_entry_cross_ent,axis=2)


def ohem_loss(logits, targets, weights):


    logits=tf.reshape(logits,shape=[-1,cfg.DATA.num_class])
    targets = tf.reshape(targets, shape=[-1])

    weights=tf.reshape(weights,shape=[-1])


    dtype = logits.dtype

    pmask = weights
    fpmask = tf.cast(pmask, dtype)
    n_positives = tf.reduce_sum(fpmask)


    no_classes = tf.cast(pmask, tf.int32)

    predictions = slim.softmax(logits)


    nmask = tf.logical_not(tf.cast(pmask,tf.bool))

    fnmask = tf.cast(nmask, dtype)

    nvalues = tf.where(nmask,
                       predictions[:, 0],
                       1. - fnmask)
    nvalues_flat = tf.reshape(nvalues, [-1])
    # Number of negative entries to select.
    max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
    n_neg = tf.cast(cfg.MODEL.max_negatives_per_positive * n_positives, tf.int32) + cfg.TRAIN.batch_size

    n_neg = tf.minimum(n_neg, max_neg_entries)

    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
    max_hard_pred = -val[-1]
    # Final negative mask.
    nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
    fnmask = tf.cast(nmask, dtype)

    # Add cross-entropy loss.
    with tf.name_scope('cross_entropy_pos'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=targets)

        neg_loss = tf.reduce_sum(loss * fpmask)

    with tf.name_scope('cross_entropy_neg'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=no_classes)
        pos_loss = tf.reduce_sum(loss * fnmask)



    return neg_loss+pos_loss





