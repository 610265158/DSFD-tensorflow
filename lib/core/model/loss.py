import tensorflow as tf
import math
from train_config import config as cfg










def calculate_loss( origin_reg, origin_cls,final_reg, final_cls,boxes,labels):
    ## first shot anchors
    boxes_small = boxes[:, 1::2]
    label_small = labels[:, 1::2]
    ## first shot loss
    reg_loss, cls_loss = ssd_loss(origin_reg, origin_cls, boxes_small, label_small)

    ## second shot anchors
    boxes_norm = boxes[:, 0::2]
    label_norm = labels[:, 0::2]
    ## second shot loss

    final_reg_loss, final_cls_loss_dual = ssd_loss(final_reg, final_cls, boxes_norm, label_norm)

    reg_loss = (reg_loss + final_reg_loss)
    cls_loss = (cls_loss + final_cls_loss_dual)


    return reg_loss+cls_loss


def ssd_loss(reg_predict,cla_predict,reg_label,cla_label):



    cla_label = tf.cast(cla_label, tf.int32)

    # whether anchor is matched
    is_matched = tf.greater(cla_label, 0)
    weights = tf.cast(is_matched,tf.float32)


    # shape [batch_size, num_anchors]


    cls_losses = ohem_loss(
        cla_predict,
        cla_label,
        weights
    )


    location_losses = localization_loss(
        reg_predict,
        reg_label, weights
    )



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


def ohem_loss(logits, targets, weights):



    logits=tf.reshape(logits,shape=[-1,cfg.DATA.num_class])
    targets = tf.reshape(targets, shape=[-1])

    weights=tf.reshape(weights,shape=[-1])

    dtype = logits.dtype

    pmask = weights
    fpmask = tf.cast(pmask, dtype)
    n_positives = tf.reduce_sum(fpmask)


    no_classes = tf.cast(pmask, tf.int32)

    predictions = tf.nn.softmax(logits)

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