

import sys
sys.path.append('.')
import tensorflow as tf
import numpy as np
from train_config import config as cfg

from lib.core.anchor.anchor import CellAnchor


def get_all_anchors(max_size,stride=None, sizes=None):
    """
    Get all anchors in the largest possible image, shifted, floatbox
    Args:
        max_size(int) : h w
        stride (int): the stride of anchors.
        sizes (tuple[int]): the sizes (sqrt area) of anchors

    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.

    """
    if stride is None:
        stride = cfg.ANCHOR.ANCHOR_STRIDE
    if sizes is None:
        sizes = cfg.ANCHOR.ANCHOR_SIZES
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    cell_anchors = CellAnchor.generate_cell_anchor(
        stride,
        scales=np.array(sizes, dtype=np.float32) / stride,
        ratios=np.array(cfg.ANCHOR.ANCHOR_RATIOS, dtype=np.float32))
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)


    field_size_y = tf.cast(tf.ceil(max_size[0] / stride), tf.float32)
    field_size_x = tf.cast(tf.ceil(max_size[1] / stride), tf.float32)
    shifts_x = tf.range(0, field_size_x) * stride
    shifts_y = tf.range(0, field_size_y) * stride
    shift_x, shift_y = tf.meshgrid(shifts_x, shifts_y)

    shift_x = tf.reshape(shift_x,shape=[1,-1])
    shift_y = tf.reshape(shift_y,shape=[1,-1])

    shifts = tf.transpose(tf.concat((shift_x, shift_y, shift_x, shift_y),axis=0))
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]
    A = cell_anchors.shape[0]

    field_of_anchors = (
        tf.reshape(cell_anchors,shape=[1, A, 4]) +
        tf.transpose(tf.reshape(shifts,shape=[1, -1, 4]),(1, 0, 2)))

    field_of_anchors = tf.reshape(field_of_anchors,shape=(field_size_y, field_size_x, A, 4))

    # FSxFSxAx4
    # Many rounding happens inside the anchor code anyway
    # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))

    ##scale it to 0 - 1

    h=tf.cast(max_size[0],tf.float32)
    w=tf.cast(max_size[1],tf.float32)

    _xx0 = (field_of_anchors[:, :, :, 0:1])/w
    _xx1 = (field_of_anchors[:, :, :, 1:2])/h
    _xx2 = (field_of_anchors[:, :, :, 2:3]+1)/w
    _xx3 = (field_of_anchors[:, :, :, 3:4]+1)/h
    field_of_anchors=tf.concat([_xx0,_xx1,_xx2,_xx3],axis=3)

    return field_of_anchors

def get_all_anchors_fpn(strides=None, sizes=None,max_size=[640,640]):
    """
    Returns:
        [anchors]: each anchors is a SxSx NUM_ANCHOR_RATIOS x4 array.
    """
    if strides is None:
        strides = cfg.ANCHOR.ANCHOR_STRIDES
    if sizes is None:
        sizes = cfg.ANCHOR.ANCHOR_SIZES
    if max_size is None:
        max_size= [cfg.DATA.max_size,cfg.DATA.max_size]

    assert len(strides) == len(sizes)
    foas = []
    for stride, size in zip(strides, sizes):
        foa = get_all_anchors(stride=stride, sizes=(size,),max_size=max_size)

        foas.append(foa)

    flatten_anchors_per_level = [tf.reshape(k,shape=(-1, 4)) for k in foas]
    anchors = tf.concat(flatten_anchors_per_level, axis=0)

    ###concat them
    return anchors


if __name__=='__main__':
    import cv2
    anchors=get_all_anchors_fpn(max_size=[640,640])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        anchors=sess.run(anchors)

    anchors=np.array(anchors)
    print(anchors.shape)
    image = np.ones(shape=[cfg.DATA.max_size, cfg.DATA.max_size, 3]) * 255
    for i in range(0,anchors.shape[0]):
        box=anchors[i]
        print(int(round((box[2]-box[0])*cfg.DATA.max_size)))
        cv2.rectangle(image, (int(round(box[0]*cfg.DATA.max_size)), int(round(box[1]*cfg.DATA.max_size))),
                      (int(round(box[2]*cfg.DATA.max_size)), int(round(box[3]*cfg.DATA.max_size))), (255, 0, 0), 1)

        cv2.namedWindow('anchors',0)
        cv2.imshow('anchors',image)
        cv2.waitKey(0)