import tensorflow as tf
from train_config import config as cfg

def batch_non_max_suppression(
        boxes, scores,labels,
        score_threshold, iou_threshold,
        max_boxes):
    """
    Arguments:
        boxes: a float tensor with shape [batch_size, N, 4].
        scores: a float tensor with shape [batch_size, N].
        score_threshold: a float number.
        iou_threshold: a float number, threshold for IoU.
        max_boxes: an integer, maximum number of retained boxes.
    Returns:
        boxes: a float tensor with shape [batch_size, max_boxes, 4].
        scores: a float tensor with shape [batch_size, max_boxes].
        num_detections: an int tensor with shape [batch_size].
    """
    def fn(x):
        boxes, scores,labels = x

        # low scoring boxes are removed
        ids = tf.where(tf.greater_equal(scores, score_threshold))
        ids = tf.squeeze(ids, axis=1)
        boxes = tf.gather(boxes, ids)
        scores = tf.gather(scores, ids)
        labels = tf.gather(labels, ids)
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_boxes, iou_threshold
        )
        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(scores, selected_indices)
        labels = tf.gather(labels, selected_indices)
        num_boxes = tf.to_int32(tf.shape(boxes)[0])

        zero_padding = max_boxes - num_boxes
        boxes = tf.pad(boxes, [[0, zero_padding], [0, 0]])
        scores = tf.pad(scores, [[0, zero_padding]])
        labels = tf.pad(labels, [[0, zero_padding]],constant_values=-1)

        boxes.set_shape([max_boxes, 4])
        scores.set_shape([max_boxes])
        labels.set_shape([max_boxes])
        return boxes, scores,labels, num_boxes

    boxes, scores, labels, num_detections = tf.map_fn(
        fn, [boxes, scores,labels],
        dtype=(tf.float32, tf.float32,tf.int64, tf.int32),
        parallel_iterations=cfg.TEST.parallel_iterations,
        back_prop=False, swap_memory=False, infer_shape=True
    )
    return boxes, scores,labels, num_detections

