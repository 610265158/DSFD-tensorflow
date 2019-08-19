# -*- coding: utf-8 -*-
# File: common.py

import numpy as np
import cv2

from tensorpack.dataflow.imgaug import transform






def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def filter_boxes_inside_shape(boxes, shape):
    """
    Args:
        boxes: (nx4), float
        shape: (h, w)

    Returns:
        indices: (k, )
        selection: (kx4)
    """
    assert boxes.ndim == 2, boxes.shape
    assert len(shape) == 2, shape
    h, w = shape
    indices = np.where(
        (boxes[:, 0] >= 0) &
        (boxes[:, 1] >= 0) &
        (boxes[:, 2] <= w) &
        (boxes[:, 3] <= h))[0]
    return indices, boxes[indices, :]

#
# try:
#     import pycocotools.mask as cocomask
#
#     # Much faster than utils/np_box_ops
#     def np_iou(A, B):
#         def to_xywh(box):
#             box = box.copy()
#             box[:, 2] -= box[:, 0]
#             box[:, 3] -= box[:, 1]
#             return box
#
#         ret = cocomask.iou(
#             to_xywh(A), to_xywh(B),
#             np.zeros((len(B),), dtype=np.bool))
#         # can accelerate even more, if using float32
#         return ret.astype('float32')
#
# except ImportError:
#     from net.utils.np_box_ops import iou as np_iou  # noqa



def np_iou(boxes1, boxes2):
    def area(boxes):
        """Computes area of boxes.

    Arguments:
        boxes: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N] representing box areas.
    """

        xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return (ymax - ymin) * (xmax - xmin)

    """Computes pairwise intersection-over-union between two box collections.

      Arguments:
          boxes1: a float tensor with shape [N, 4].GT
          boxes2: a float tensor with shape [M, 4].ANCHOR
      Returns:
          a float tensor with shape [N, M] representing pairwise iou scores.
      """

    intersections = intersection(boxes1, boxes2)  #####################transfored wrong

    areas1 = area(boxes1)
    areas2 = area(boxes2)
    unions = np.expand_dims(areas1, 1) + np.expand_dims(areas2, 0) - intersections

    return np.clip(intersections / unions, 0.0, 1.0)
def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.

  Arguments:
      boxes1: a float tensor with shape [N, 4].
      boxes2: a float tensor with shape [M, 4].
  Returns:
      a float tensor with shape [N, M] representing pairwise intersections.
  """
    ##########np transformed wrong need review
    xmin1, ymin1, xmax1, ymax1 = np.split(boxes1, indices_or_sections=4, axis=1)
    xmin2, ymin2, xmax2, ymax2 = np.split(boxes2, indices_or_sections=4, axis=1)
    # they all have shapes like [None, 1]

    all_pairs_min_ymax = np.minimum(ymax1, np.transpose(ymax2))
    all_pairs_max_ymin = np.maximum(ymin1, np.transpose(ymin2))

    intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(xmax1, np.transpose(xmax2))
    all_pairs_max_xmin = np.maximum(xmin1, np.transpose(xmin2))
    intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    # they all have shape [N, M]
    return intersect_heights * intersect_widths
