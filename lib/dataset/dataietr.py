


import os
import random
import cv2
import numpy as np
import traceback

from lib.helper.logger import logger
from tensorpack.dataflow import DataFromGenerator
from tensorpack.dataflow import BatchData, MultiProcessPrefetchData


from lib.dataset.augmentor.augmentation import Random_scale_withbbox,\
                                                Random_flip,\
                                                baidu_aug,\
                                                dsfd_aug,\
                                                Fill_img

from lib.dataset.augmentor.visual_augmentation import ColorDistort

from train_config import config as cfg


class data_info():
    def __init__(self,img_root,txt):
        self.txt_file=txt
        self.root_path = img_root
        self.metas=[]


        self.read_txt()

    def read_txt(self):
        with open(self.txt_file) as _f:
            txt_lines=_f.readlines()
        txt_lines.sort()
        for line in txt_lines:
            line=line.rstrip()

            _img_path=line.rsplit('| ',1)[0]
            _label=line.rsplit('| ',1)[-1]

            current_img_path=os.path.join(self.root_path,_img_path)
            current_img_label=_label
            self.metas.append([current_img_path,current_img_label])

            ###some change can be made here
        logger.info('the dataset contains %d images'%(len(txt_lines)))
        logger.info('the datasets contains %d samples'%(len(self.metas)))


    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas

class MutiScaleBatcher(BatchData):

    def __init__(self, ds, batch_size, remainder=False, use_list=False,scale_range=None,input_size=(512,512),divide_size=32):
        """
        Args:
            ds (DataFlow): A dataflow that produces either list or dict.
                When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guaranteed to have the same batch size.
                If set to True, `len(ds)` must be accurate.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
        """
        super(BatchData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= len(ds)
            except NotImplementedError:
                pass
        self.batch_size = int(batch_size)
        self.remainder = remainder
        self.use_list = use_list

        self.scale_range=scale_range
        self.divide_size=divide_size

        self.input_size=input_size

    def __iter__(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """

        ##### pick a scale and shape aligment

        holder = []
        for data in self.ds:

            image,boxes_,klass_=data[0],data[1],data[2]


            ###cove the small faces
            boxes_clean = []
            for i in range(boxes_.shape[0]):
                box = boxes_[i]

                if (box[3] - box[1]) < cfg.DATA.cover_small_face or (box[2] - box[0]) < cfg.DATA.cover_small_face:
                    image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] = np.array(cfg.DATA.PIXEL_MEAN, dtype=image.dtype)
                    continue
                else:
                    boxes_clean.append(box)


            boxes_=np.array(boxes_)

            data=[image,boxes_,klass_]
            holder.append(data)
            if len(holder) == self.batch_size:
                target = self.produce_target(holder)



                yield BatchData.aggregate_batch(target, self.use_list)
                del holder[:]

        if self.remainder and len(holder) > 0:
            yield BatchData._aggregate_batch(holder, self.use_list)



    def produce_target(self,holder):
        alig_data = []

        if self.scale_range is not None:
            max_shape = [random.randint(*self.scale_range),random.randint(*self.scale_range)]

            max_shape[0] = int(np.ceil(max_shape[0] / self.divide_size) * self.divide_size)
            max_shape[1] = int(np.ceil(max_shape[1] / self.divide_size) * self.divide_size)

            cfg.ANCHOR.achor.reset_anchors((max_shape[1], max_shape[0]))
        else:
            max_shape=self.input_size

        # copy images to the upper left part of the image batch object
        for [image, boxes_, klass_] in holder:

            # construct an image batch object
            image, shift_x, shift_y = Fill_img(image, target_width=max_shape[0], target_height=max_shape[1])
            boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
            h, w, _ = image.shape
            boxes_[:, 0] /= w
            boxes_[:, 1] /= h
            boxes_[:, 2] /= w
            boxes_[:, 3] /= h
            image = image.astype(np.uint8)

            image = cv2.resize(image, (max_shape[0], max_shape[1]))

            boxes_[:, 0] *= max_shape[0]
            boxes_[:, 1] *= max_shape[1]
            boxes_[:, 2] *= max_shape[0]
            boxes_[:, 3] *= max_shape[1]


            if cfg.TRAIN.vis:
                for __box in boxes_:
                    cv2.rectangle(image, (int(__box[0]), int(__box[1])),
                                  (int(__box[2]), int(__box[3])), (255, 0, 0), 4)

            all_boxes, all_labels = cfg.ANCHOR.achor.produce_target(boxes_, klass_)

            alig_data.append([image, all_boxes, all_labels])

        return alig_data


class DsfdDataIter():

    def __init__(self, img_root_path='', ann_file=None, training_flag=True, shuffle=True):

        self.color_augmentor = ColorDistort()

        self.training_flag = training_flag

        self.lst = self.parse_file(img_root_path, ann_file)

        self.shuffle = shuffle

    def __iter__(self):
        idxs = np.arange(len(self.lst))

        while True:
            if self.shuffle:
                np.random.shuffle(idxs)
            for k in idxs:
                yield self._map_func(self.lst[k], self.training_flag)



    def parse_file(self,im_root_path,ann_file):
        '''
        :return: [fname,lbel]     type:list
        '''
        logger.info("[x] Get dataset from {}".format(im_root_path))

        ann_info = data_info(im_root_path, ann_file)
        all_samples = ann_info.get_all_sample()

        return all_samples

    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here
        try:
            fname, annos = dp
            image = cv2.imread(fname, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            labels = annos.split(' ')
            boxes = []


            for label in labels:
                bbox = np.array(label.split(','), dtype=np.float)
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])

            boxes = np.array(boxes, dtype=np.float)


            if is_training:

                sample_dice = random.uniform(0, 1)
                if sample_dice > 0.8 and sample_dice <= 1:
                    image, boxes = Random_scale_withbbox(image, boxes, target_shape=[cfg.DATA.hin, cfg.DATA.win],
                                                         jitter=0.3)
                elif sample_dice > 0.4 and sample_dice <= 0.8:
                    boxes_ = boxes[:, 0:4]
                    klass_ = boxes[:, 4:]

                    image, boxes_, klass_ = dsfd_aug(image, boxes_, klass_)

                    image = image.astype(np.uint8)
                    boxes = np.concatenate([boxes_, klass_], axis=1)
                else:
                    boxes_ = boxes[:, 0:4]
                    klass_ = boxes[:, 4:]
                    image, boxes_, klass_ = baidu_aug(image, boxes_, klass_)

                    image = image.astype(np.uint8)
                    boxes = np.concatenate([boxes_, klass_], axis=1)

                if random.uniform(0, 1) > 0.5:
                    image, boxes = Random_flip(image, boxes)

                if random.uniform(0, 1) > 0.5:
                    image =self.color_augmentor(image)

            else:
                boxes_ = boxes[:, 0:4]
                klass_ = boxes[:, 4:]
                image, shift_x, shift_y = Fill_img(image, target_width=cfg.DATA.win, target_height=cfg.DATA.hin)
                boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
                h, w, _ = image.shape
                boxes_[:, 0] /= w
                boxes_[:, 1] /= h
                boxes_[:, 2] /= w
                boxes_[:, 3] /= h
                image = image.astype(np.uint8)
                image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin))

                boxes_[:, 0] *= cfg.DATA.win
                boxes_[:, 1] *= cfg.DATA.hin
                boxes_[:, 2] *= cfg.DATA.win
                boxes_[:, 3] *= cfg.DATA.hin
                image = image.astype(np.uint8)
                boxes = np.concatenate([boxes_, klass_], axis=1)




            if boxes.shape[0] == 0 or np.sum(image) == 0:
                boxes_ = np.array([[0, 0, 100, 100]])
                klass_ = np.array([0])
            else:
                boxes_ = np.array(boxes[:, 0:4], dtype=np.float32)
                klass_ = np.array(boxes[:, 4], dtype=np.int64)




        except:
            logger.warn('there is an err with %s' % fname)
            traceback.print_exc()
            image = np.zeros(shape=(cfg.DATA.hin, cfg.DATA.win, 3), dtype=np.float32)
            boxes_ = np.array([[0, 0, 100, 100]])
            klass_ = np.array([0])


        return image, boxes_, klass_


class DataIter():
    def __init__(self, img_root_path='', ann_file=None, training_flag=True):

        self.shuffle = True
        self.training_flag = training_flag

        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size

        self.generator = DsfdDataIter(img_root_path, ann_file, self.training_flag )

        self.ds = self.build_iter()



    def parse_file(self, im_root_path, ann_file):

        raise NotImplementedError("you need implemented the parse func for your data")

    def build_iter(self,):


        ds = DataFromGenerator(self.generator)


        if cfg.DATA.mutiscale and self.training_flag:
            ds = MutiScaleBatcher(ds, self.num_gpu * self.batch_size, scale_range=cfg.DATA.scales,
                                  input_size=(cfg.DATA.hin, cfg.DATA.win))
        else:
            ds = MutiScaleBatcher(ds, self.num_gpu * self.batch_size, input_size=(cfg.DATA.hin, cfg.DATA.win))

        ds = MultiProcessPrefetchData(ds, self.prefetch_size, self.process_num)
        ds.reset_state()
        ds = ds.get_data()
        return ds


    def __next__(self):
        return next(self.ds)

