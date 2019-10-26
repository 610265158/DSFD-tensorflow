from lib.helper.logger import logger
from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import DataIter

from lib.core.model.light.dsfd import DSFD as lightnet_dsfd
from lib.core.model.vgg.dsfd import DSFD as vgg_dsfd

import tensorflow as tf
import cv2
import numpy as np

from train_config import config as cfg
import setproctitle

logger.info('The trainer start')

setproctitle.setproctitle("dsfd")

def main():

    epochs=cfg.TRAIN.epoch
    batch_size=cfg.TRAIN.batch_size

    enable_function=False

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    devices = ['/device:GPU:{}'.format(i) for i in range(cfg.TRAIN.num_gpu)]

    strategy = tf.distribute.MirroredStrategy(devices)
    with strategy.scope():

        if 'vgg' in cfg.MODEL.net_structure:
            model=vgg_dsfd()
        elif 'Lightnet' in cfg.MODEL.net_structure:
            model = lightnet_dsfd()

        ###run a time to build the model
        image = np.zeros(shape=(1, 512, 512, 3), dtype=np.float32)
        model.inference(image)


    ## load pretrained weights
    if cfg.MODEL.pretrained_model is not None:
        logger.info('load pretrained params from %s'%cfg.MODEL.pretrained_model)
        model.load_weights(cfg.MODEL.pretrained_model)

    ### build trainer
    trainer = Train(epochs, enable_function, model, batch_size, strategy)

    ### build dataiter
    train_ds = DataIter(cfg.DATA.root_path, cfg.DATA.train_txt_path, True)
    test_ds = DataIter(cfg.DATA.root_path, cfg.DATA.val_txt_path, False)


    ### it's a tensorpack data iter, produce a batch every iter
    train_dataset=tf.data.Dataset.from_generator(train_ds,
                                                 output_types=(tf.float32,tf.float32,tf.float32),
                                                 output_shapes=([None,None,None,None],[None,None,None],[None,None]))
    test_dataset = tf.data.Dataset.from_generator(test_ds,
                                                  output_types=(tf.float32,tf.float32,tf.float32),
                                                  output_shapes=([None,None,None,None],[None,None,None],[None,None]))

    ####
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


    ## check the data
    if cfg.TRAIN.vis:
        for images,labels,matches in train_dist_dataset:
            #images,labels,matches=one_batch
            print(images)
            for i in range(images.shape[0]):
                example_image=np.array(images[i],dtype=np.uint8)
                example_label=np.array(labels[i])


                cv2.imshow('example',example_image)
                cv2.waitKey(0)



    ##train
    trainer.custom_loop(train_dist_dataset,
                        test_dist_dataset,
                        strategy)

if __name__=='__main__':
    main()