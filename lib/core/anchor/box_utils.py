

import sys
sys.path.append('.')
import tensorflow as tf
import numpy as np
# a small value
EPSILON = 1e-8
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]

"""
Tools for dealing with bounding boxes.
All boxes are of the format [ymin, xmin, ymax, xmax]
if not stated otherwise.
And box coordinates are normalized to [0, 1] range.
"""

def iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between two box collections.

    Arguments:
        boxes1: a float tensor with shape [N, 4].GT
        boxes2: a float tensor with shape [M, 4].ANCHOR
    Returns:
        a float tensor with shape [N, M] representing pairwise iou scores.
    """

    intersections = intersection(boxes1, boxes2)#####################transfored wrong

    areas1 = area(boxes1)
    areas2 = area(boxes2)
    unions = np.expand_dims(areas1, 1) + np.expand_dims(areas2, 0) - intersections

    return np.clip(intersections/unions, 0.0, 1.0)


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.

    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise intersections.
    """
    ##########np transformed wrong need review
    ymin1, xmin1, ymax1, xmax1 = np.split(boxes1, indices_or_sections=4, axis=1)
    ymin2, xmin2, ymax2, xmax2 = np.split(boxes2, indices_or_sections=4, axis=1)
    # they all have shapes like [None, 1]

    all_pairs_min_ymax = np.minimum(ymax1, np.transpose(ymax2))
    all_pairs_max_ymin = np.maximum(ymin1, np.transpose(ymin2))

    intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(xmax1, np.transpose(xmax2))
    all_pairs_max_xmin = np.maximum(xmin1, np.transpose(xmin2))
    intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    # they all have shape [N, M]
    return intersect_heights * intersect_widths


def area(boxes):
    """Computes area of boxes.

    Arguments:
        boxes: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N] representing box areas.
    """

    ymin, xmin, ymax, xmax = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    return (ymax - ymin) * (xmax - xmin)


def to_minmax_coordinates(boxes):
    """Convert bounding boxes of the format
    [cy, cx, h, w] to the format [ymin, xmin, ymax, xmax].

    Arguments:
        boxes: a list of float tensors with shape [N]
            that represent cy, cx, h, w.
    Returns:
        a list of float tensors with shape [N]
        that represent ymin, xmin, ymax, xmax.
    """

    cy, cx, h, w = boxes
    ymin, xmin = cy - 0.5*h, cx - 0.5*w
    ymax, xmax = cy + 0.5*h, cx + 0.5*w
    return [ymin, xmin, ymax, xmax]


def to_center_coordinates(boxes):
    """Convert bounding boxes of the format
    [ymin, xmin, ymax, xmax] to the format [cy, cx, h, w].

    Arguments:
        boxes: a list of float tensors with shape [N]
            that represent ymin, xmin, ymax, xmax.
    Returns:
        a list of float tensors with shape [N]
        that represent cy, cx, h, w.
    """

    #ymin, xmin, ymax, xmax = boxes
    xmin, ymin, xmax, ymax = boxes
    h = ymax - ymin
    w = xmax - xmin
    cy = ymin + 0.5 * h
    cx = xmin + 0.5 * w
    return [cy, cx, h, w]


def encode(boxes, anchors):

    # print(boxes)
    # print(anchors)
    """Encode boxes with respect to anchors.

    Arguments:
        boxes: a float tensor with shape [N, 4].
        anchors: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N, 4],
        anchor-encoded boxes of the format [ty, tx, th, tw].
    """

    ycenter_a, xcenter_a, ha, wa = to_center_coordinates([anchors[:,0],anchors[:,1],anchors[:,2],anchors[:,3]])
    ycenter, xcenter, h, w = to_center_coordinates([boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]])

    # to avoid NaN in division and log below
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = np.log(w / wa)
    th = np.log(h / ha)

    ty *= SCALE_FACTORS[0]
    tx *= SCALE_FACTORS[1]
    th *= SCALE_FACTORS[2]
    tw *= SCALE_FACTORS[3]

    return np.stack([ty, tx, th, tw], axis=1)



def decode(codes, anchors):
    """Decode relative codes to boxes.

    Arguments:
        codes: a float tensor with shape [N, 4],
            anchor-encoded boxes of the format [ty, tx, th, tw].
        anchors: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N, 4],
        bounding boxes of the format [ymin, xmin, ymax, xmax].
    """
    with tf.name_scope('decode_predictions'):

        ycenter_a, xcenter_a, ha, wa = to_center_coordinates(tf.unstack(anchors, axis=1))
        ty, tx, th, tw = tf.unstack(codes, axis=1)

        ty /= SCALE_FACTORS[0]
        tx /= SCALE_FACTORS[1]
        th /= SCALE_FACTORS[2]
        tw /= SCALE_FACTORS[3]
        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a

        return tf.stack(to_minmax_coordinates([ycenter, xcenter, h, w]), axis=1)


def batch_decode(box_encodings, anchors):
    """Decodes a batch of box encodings with respect to the anchors.

    Arguments:
        box_encodings: a float tensor with shape [batch_size, num_anchors, 4].
        anchors: a float tensor with shape [num_anchors, 4].
    Returns:
        a float tensor with shape [batch_size, num_anchors, 4].
        It contains the decoded boxes.
    """
    batch_size = tf.shape(box_encodings)[0]
    num_anchors = tf.shape(box_encodings)[1]

    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchors, 0),
        [batch_size, 1, 1]
    )  # shape [batch_size, num_anchors, 4]
    decoded_boxes = decode(
        tf.reshape(box_encodings, [-1, 4]),
        tf.reshape(tiled_anchor_boxes, [-1, 4])
    )  # shape [batch_size * num_anchors, 4]

    decoded_boxes = tf.reshape(
        decoded_boxes,
        [batch_size, num_anchors, 4]
    )
    decoded_boxes = tf.clip_by_value(decoded_boxes, 0.0, 1.0)
    return decoded_boxes
