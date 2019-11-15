import tensorflow as tf
from train_config import config as cfg

from tensorflow_core.python.keras.applications.mobilenet_v2 import mobilenet_v2


class MobileNet(tf.keras.Model):
    def __init__(self,
                 model_size=1.0,
                 weights='imagenet'):
        super(MobileNet, self).__init__()

        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(cfg.DATA.hin, cfg.DATA.win, 3),
                                                                    include_top=False,
                                                                    alpha=model_size,
                                                                    weights=weights)

        base_model.summary()

        layers_out = ["block_6_expand_relu", "block_13_expand_relu", "block_16_expand_relu"]
        intermid_outputs = [base_model.get_layer(layer_name).output for layer_name in layers_out]

        self.backbone_features = tf.keras.Model(inputs=base_model.input, outputs=intermid_outputs)

        self.backbone_features.summary()

    def call(self, inputs, training):
        p1, p2,p3 = self.backbone_features(inputs, training=training)

        return p1, p2,p3