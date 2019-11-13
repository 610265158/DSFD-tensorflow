import tensorflow as tf

from lib.core.model.light.dsfd import DSFD
from train_config import config as cfg
from lib.core.anchor.tf_anchors import get_all_anchors_fpn
from lib.core.anchor.box_utils import batch_decode


class DSFDLite(DSFD):
    #below are funcs for tflite converter
    def __init__(self,input_shape):
        super(DSFDLite, self).__init__()
        '''

        :param input_shape: h,w,c or h,w
        '''
        self.input_size=input_shape


        self.pre_define_anchor=self.get_pre_define_anchors(self.input_size[0],self.input_size[1])

    def inference(self, images,
                        score_threshold=cfg.TEST.score_thres, \
                        iou_threshold=cfg.TEST.iou_thres):
        '''
        redefine the func to disable it, because tflite now only support one method per model
        :param images:
        :param score_threshold:
        :param iou_threshold:
        :return:
        '''
        pass



    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
    def inference_fixed(self, images):

        x = self.preprocess(images)

        of1, of2, of3 = self.base_model(x, training=False)



        fms = [ of1, of2, of3]



        if cfg.MODEL.fpn:
            fpn_fms = self.fpn(fms, training = False)
        else:
            fpn_fms=fms


        if cfg.MODEL.cpm:
            for i in range(len(fpn_fms)):
                fpn_fms[i] = self.cpm_ops[i](fpn_fms[i], training=False)


        fpn_reg, fpn_cls = self.ssd_head_fem(fpn_fms, training=False)

        ###get anchor
        ###### adjust the anchors to the image shape, but it trains with a fixed h,w

        boxes = batch_decode(fpn_reg, self.pre_define_anchor)

        # it has shape [batch_size, num_anchors, 4]

        scores = tf.nn.softmax(fpn_cls, axis=2)[:, :, 1:]  ##ignore the bg
        # it has shape [batch_size, num_anchors,class]
        labels = tf.argmax(scores, axis=2)
        # it has shape [batch_size, num_anchors]

        scores = tf.reduce_max(scores, axis=2)
        scores =tf.expand_dims(scores,axis=-1)
        # it has shape [batch_size, num_anchors]

        res=tf.concat([boxes,scores],axis=2)


        return res
    def get_pre_define_anchors(self,h,w):
        anchors_ = get_all_anchors_fpn(max_size=[h, w])

        if cfg.MODEL.dual_mode:
            anchors_ = anchors_[0::2]
        else:
            anchors_ = anchors_
        return anchors_






if __name__=='__main__':


    model=DSFDLite((320,320))
    model.load_weights(cfg.MODEL.pretrained_model)