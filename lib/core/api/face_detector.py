import tensorflow as tf
import numpy as np
import cv2
import time

from train_config import config as cfg












class FaceDetector:
    def __init__(self, model_path):
        """
        Arguments:
            model_path: a string, path to a pb file.
        """
        self._graph = tf.Graph()

        with self._graph.as_default():
            self._graph, self._sess = self.init_model(model_path)

            self.input_image = tf.get_default_graph().get_tensor_by_name('tower_0/images:0')
            self.training = tf.get_default_graph().get_tensor_by_name('training_flag:0')
            self.output_ops = [
                tf.get_default_graph().get_tensor_by_name('tower_0/boxes:0'),
                tf.get_default_graph().get_tensor_by_name('tower_0/labels:0'),
                tf.get_default_graph().get_tensor_by_name('tower_0/scores:0'),
                tf.get_default_graph().get_tensor_by_name('tower_0/num_detections:0'),
            ]


    def __call__(self, image, score_threshold=0.5,minface=40):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            scores: a float numpy array of shape [num_faces].

        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """



        h, w, _ = image.shape

        input_shape=((w//32+1)*32,(h//32+1)*32)
        image_fornet=cv2.resize(image,input_shape)

        image_fornet = np.expand_dims(image_fornet, 0)

        start = time.time()
        boxes,labels, scores, num_boxes = self._sess.run(
            self.output_ops, feed_dict={self.input_image: image_fornet,self.training:False}
        )
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        labels = labels[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]
        labels = labels[to_keep]
        ###recorver to raw image
        scaler = np.array([h, w, h, w], dtype='float32')
        boxes = boxes * scaler
        #boxes =boxes-np.array([shift_y, shift_x, shift_y, shift_x], dtype='float32')

        scores=np.expand_dims(scores, 0).reshape([-1,1])
        labels = np.expand_dims(labels, 0).reshape([-1, 1])
        for i in range(boxes.shape[0]):
            boxes[i] = np.array([boxes[i][1], boxes[i][0], boxes[i][3],boxes[i][2]])  #####the faceboxe produce ymin,xmin,ymax,xmax


        return np.concatenate([boxes, scores],axis=1)
    def Fill_img(self,img_raw,target_height,target_width,label=None):

        ###sometimes use in objs detects
        channel=img_raw.shape[2]
        raw_height = img_raw.shape[0]
        raw_width = img_raw.shape[1]
        if raw_width / raw_height >= target_width / target_height:
            shape_need = [int(target_height / target_width * raw_width), raw_width, channel]
            img_fill = np.zeros(shape_need, dtype=img_raw.dtype)
            shift_x=(img_fill.shape[1]-raw_width)//2
            shift_y=(img_fill.shape[0]-raw_height)//2
            for i in range(channel):
                img_fill[shift_y:raw_height+shift_y, shift_x:raw_width+shift_x, i] = img_raw[:,:,i]
        else:
            shape_need = [raw_height, int(target_width / target_height * raw_height), channel]
            img_fill = np.zeros(shape_need, dtype=img_raw.dtype)
            shift_x = (img_fill.shape[1] - raw_width) // 2
            shift_y = (img_fill.shape[0] - raw_height) // 2
            for i in range(channel):
                img_fill[shift_y:raw_height + shift_y, shift_x:raw_width + shift_x, i] = img_raw[:, :, i]
        if label is None:
            return img_fill,shift_x,shift_y
        else:
            label[:,0]+=shift_x
            label[:, 1]+=shift_y
            return img_fill,label
    def init_model(self,args):

        if len(args) == 1:
            use_pb = True
            pb_path = args[0]
        else:
            use_pb = False
            meta_path = args[0]
            restore_model_path = args[1]

        def ini_ckpt():
            graph = tf.Graph()
            graph.as_default()
            configProto = tf.ConfigProto()
            configProto.gpu_options.allow_growth = True
            sess = tf.Session(config=configProto)
            # load_model(model_path, sess)
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, restore_model_path)

            print("Model restred!")
            return (graph, sess)

        def init_pb(model_path):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            compute_graph = tf.Graph()
            compute_graph.as_default()
            sess = tf.Session(config=config)
            with tf.gfile.GFile(model_path, 'rb') as fid:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

            # saver = tf.train.Saver(tf.global_variables())
            # saver.save(sess, save_path='./tmp.ckpt')
            return (compute_graph, sess)

        if use_pb:
            model = init_pb(pb_path)
        else:
            model = ini_ckpt()

        graph = model[0]
        sess = model[1]

        return graph, sess
