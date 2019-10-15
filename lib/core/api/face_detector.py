import tensorflow as tf
import numpy as np
import cv2
import time

from train_config import config as cfg

class FaceDetector:
    def __init__(self, model_path):
        """
        Arguments:
            model_path: a string, path to the model params file.
        """

        self.model = tf.saved_model.load(model_path)

    def __call__(self, image,score_threshold=0.5,input_shape=(320,320)):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            input_shape: (h,w)
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 5].

        """

        if input_shape is None:
            h,w,c=image.shape
            input_shape = ((h // 32 + 1) * 32, (w // 32 + 1) * 32)

        image_fornet, scale_x, scale_y = self.preprocess(image,
                                                         target_height=input_shape[0],
                                                         target_width =input_shape[1])

        image_fornet = np.expand_dims(image_fornet, 0)

        start = time.time()
        res = self.model.inference(image_fornet)

        print('xx', time.time() - start)

        boxes = res['boxes'].numpy()
        label = res['labels'].numpy()
        scores = res['scores'].numpy()
        num_boxes = res['num_boxes'].numpy()

        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        label= label[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]
        label= label[to_keep]
        ###recorver to raw image
        scaler = np.array([input_shape[0] / scale_y,
                           input_shape[1] / scale_x,
                           input_shape[0] / scale_y,
                           input_shape[1] / scale_x], dtype='float32')
        boxes = boxes * scaler

        scores = np.expand_dims(scores, 0).reshape([-1, 1])

        #####the tf.nms produce ymin,xmin,ymax,xmax,  swap it in to xmin,ymin,xmax,ymax
        for i in range(boxes.shape[0]):
            boxes[i] = np.array([boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]])
        return np.concatenate([boxes, scores], axis=1)





    def preprocess(self, image, target_height, target_width, label=None):

        ###sometimes use in objs detects
        h, w, c = image.shape

        bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype) + np.array(cfg.DATA.PIXEL_MEAN,
                                                                                                dtype=image.dtype)

        long_side = max(h, w)

        scale_x = scale_y = target_height / long_side

        image = cv2.resize(image, None, fx=scale_x, fy=scale_y)

        h_, w_, _ = image.shape
        bimage[:h_, :w_, :] = image

        return bimage, scale_x, scale_y

