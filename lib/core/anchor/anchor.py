import numpy as np



import sys
sys.path.append('.')


from lib.core.anchor.box_utils import encode,np_iou

from train_config import config as cfg



class CellAnchor():

    def __init__(self):
      pass

    @classmethod
    def generate_cell_anchor(self,base_size=16,ratios=[0.5,1.,2.],scales=2**np.arange(3,6),rect=cfg.ANCHOR.rect):
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        anchors_in_ratios = self.make_anchor_in_ratios(base_anchor, ratios, rect)
        anchors_in_scales = self.make_anchor_in_sclaes(anchors_in_ratios, scales)
        return anchors_in_scales

    @classmethod
    def _to_whxy(self,anchors):
        w=anchors[2]-anchors[0]+1
        h=anchors[3]-anchors[1]+1

        x=anchors[0]+(w-1)/2
        y=anchors[1]+(h-1)/2
        return w,h,x,y

    @classmethod
    def _to_xyxy(self,w,h,x,y):

        x0=x-(w-1)/2
        y0=y-(h-1)/2
        x1=x+(w-1)/2
        y1 = y + (h-1) / 2

        return np.stack((x0,y0,x1,y1),axis=-1)

    @classmethod
    def make_anchor_in_ratios(self,base_anchor,ratios,rect=False):

        anchors_in_ratios=[]
        w,h,x,y=self._to_whxy(base_anchor)
        area=w*h

        for ratio in ratios:

            ### choose the face anchor ratio h/w ==1.5 or 1
            if rect:
                w=h=np.round(np.sqrt(area/ratio))
                if cfg.ANCHOR.rect_longer:
                    h=np.round(1.5*w)
            else:
                w=np.round(np.sqrt(area/ratio))
                h=np.round(ratio*w)

            anchors_in_ratios.append(self._to_xyxy(w,h,x,y))


        return np.array(anchors_in_ratios)

    @classmethod
    def make_anchor_in_sclaes(self,anchors,scales):
        anchors_res=[]

        for anchor in anchors:
            w,h,x,y=self._to_whxy(anchor)
            w=w*scales
            h=h*scales
            anchors_sclase=self._to_xyxy(w,h,x,y)
            anchors_res.append(anchors_sclase)
        return np.array(anchors_res).reshape([-1,4])

class Anchor():

    def __init__(self):

        self.strides=cfg.ANCHOR.ANCHOR_STRIDES
        self.sizes = cfg.ANCHOR.ANCHOR_SIZES

        self.ratios=cfg.ANCHOR.ANCHOR_RATIOS


        self.max_size=cfg.DATA.max_size     ##use to calculate the anchor

        self.anchors=self.produce_anchors()


    def produce_anchors(self):
        anchors_per_level = self.get_all_anchors_fpn()
        flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
        all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)
        return  all_anchors_flatten

    def get_all_anchors(self,stride=None, sizes=None):
        """
        Get all anchors in the largest possible image, shifted, floatbox
        Args:
            stride (int): the stride of anchors.
            sizes (tuple[int]): the sizes (sqrt area) of anchors

        Returns:
            anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
            The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.

        """

        # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
        # are centered on stride / 2, have (approximate) sqrt areas of the specified
        # sizes, and aspect ratios as given.
        cell_anchors = CellAnchor.generate_cell_anchor(
            stride,
            scales=np.array(sizes, dtype=np.float) / stride,
            ratios=np.array(self.ratios, dtype=np.float))
        # anchors are intbox here.
        # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

        field_size_y = int(np.ceil(self.max_size[0] / stride))
        field_size_x = int(np.ceil(self.max_size[1] / stride))

        shifts_x = np.arange(0, field_size_x) * stride
        shifts_y = np.arange(0, field_size_y) * stride
        shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
        shift_x = shift_x.flatten()
        shift_y = shift_y.flatten()
        shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
        # Kx4, K = field_size * field_size
        K = shifts.shape[0]

        A = cell_anchors.shape[0]
        field_of_anchors = (
                cell_anchors.reshape((1, A, 4)) +
                shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        field_of_anchors = field_of_anchors.reshape((field_size_y, field_size_x, A, 4))
        # FSxFSxAx4
        # Many rounding happens inside the anchor code anyway
        # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
        field_of_anchors = field_of_anchors.astype('float32')
        field_of_anchors[:, :, :, [2, 3]] += 1
        return field_of_anchors

    def get_all_anchors_fpn(self):
        """
        Returns:
            [anchors]: each anchors is a SxSx NUM_ANCHOR_RATIOS x4 array.
        """
        strides =self.strides
        sizes = self.sizes

        assert len(strides) == len(sizes)
        foas = []
        for stride, size in zip(strides, sizes):
            foa = self.get_all_anchors(stride=stride, sizes=(size,))
            foas.append(foa)


        return foas

    def produce_target(self, boxes, labels):
        boxes = boxes.copy()

        all_anchors_flatten =self.anchors

        # inside_ind, inside_anchors = filter_boxes_inside_shape(all_anchors_flatten, image.shape[:2])
        inside_anchors = all_anchors_flatten

        # obtain anchor labels and their corresponding gt boxes
        anchor_labels, anchor_gt_boxes = self.get_anchor_labels(inside_anchors, boxes, labels)

        # start = 0
        # multilevel_inputs = []
        # for level_anchor in anchors_per_level:
        #     assert level_anchor.shape[2] == len(cfg.ANCHOR.ANCHOR_RATIOS)
        #     anchor_shape = level_anchor.shape[:3]   # fHxfWxNUM_ANCHOR_RATIOS
        #     num_anchor_this_level = np.prod(anchor_shape)
        #     end = start + num_anchor_this_level
        #     multilevel_inputs.append(
        #         (all_labels[start: end].reshape(anchor_shape),
        #          all_boxes[start: end, :].reshape(anchor_shape + (4,))
        #          ))
        #     start = end
        # assert end == num_all_anchors, "{} != {}".format(end, num_all_anchors)
        # return multilevel_inputs
        return anchor_gt_boxes, anchor_labels

    def get_anchor_labels(self,anchors, gt_boxes, labels):
        # This function will modify labels and return the filtered inds

        NA, NB = len(anchors), len(gt_boxes)
        assert NB > 0  # empty images should have been filtered already
        # ##########
        anchor_matched_already = np.zeros((NA,), dtype='int32')
        gt_boxes_mathed_already = np.zeros((NB,), dtype='int32')
        anchor_labels = np.zeros((NA,), dtype='int32')
        anchor_boxes = np.zeros((NA, 4), dtype='float32')

        box_ious = np_iou(anchors, gt_boxes)  # NA x NB

        # for each anchor box choose the groundtruth box with largest iou
        max_iou = box_ious.max(axis=1)  # NA
        positive_anchor_indices = np.where(max_iou > cfg.ANCHOR.POSITIVE_ANCHOR_THRESH)[0]
        # negative_anchor_indices = np.where(max_iou < cfg.ANCHOR.NEGATIVE_ANCHOR_THRESH)[0]

        positive_iou = box_ious[positive_anchor_indices]
        matched_gt_box_indices = positive_iou.argmax(axis=1)

        anchor_labels[positive_anchor_indices] = labels[matched_gt_box_indices]
        anchor_boxes[positive_anchor_indices] = gt_boxes[matched_gt_box_indices]
        anchor_matched_already[positive_anchor_indices] = 1  #### marked as matched
        gt_boxes_mathed_already[matched_gt_box_indices] = 1  #### marked as matched

        if np.sum(anchor_matched_already) > 0:
            n = np.sum(anchor_matched_already) / np.sum(gt_boxes_mathed_already)
        else:
            n = cfg.ANCHOR.AVG_MATCHES
        n = n if n > cfg.ANCHOR.AVG_MATCHES else cfg.ANCHOR.AVG_MATCHES
        if not cfg.ANCHOR.super_match:
            n = cfg.ANCHOR.AVG_MATCHES
        # some gt_boxes may not matched, find them and match them with n anchors for each gt box
        box_ious[box_ious < cfg.ANCHOR.NEGATIVE_ANCHOR_THRESH] = 0
        sorted_ious = np.argsort(-box_ious, axis=0)

        sorted_ious = sorted_ious[np.logical_not(anchor_matched_already)]

        for i in range(0, len(gt_boxes_mathed_already)):
            matched_count = np.sum(matched_gt_box_indices == gt_boxes_mathed_already[i])

            if matched_count >= n:
                continue
            else:
                for j in range(0, int(n - matched_count)):
                    if box_ious[sorted_ious[j][i]][i] > cfg.ANCHOR.NEGATIVE_ANCHOR_THRESH:
                        anchor_labels[sorted_ious[j][i]] = labels[i]
                        anchor_boxes[sorted_ious[j][i]] = gt_boxes[i]

                        anchor_matched_already[sorted_ious[j][i]] = 1

                        gt_boxes_mathed_already[i] = 1

        fg_boxes = anchor_boxes[anchor_matched_already.astype(np.bool)]

        matched_anchors = anchors[anchor_matched_already.astype(np.bool)]

        ##select and normlised the box coordinate
        fg_boxes[:,0::2] = fg_boxes[:,0::2] / self.max_size[0]
        fg_boxes[:, 1::2] = fg_boxes[:, 1::2] / self.max_size[1]

        matched_anchors[:,0::2] = matched_anchors[:,0::2] / self.max_size[0]
        matched_anchors[:, 1::2] = matched_anchors[:, 1::2] / self.max_size[1]



        encode_fg_boxes = encode(fg_boxes, matched_anchors)
        anchor_boxes[anchor_matched_already.astype(np.bool)] = encode_fg_boxes
        # assert len(fg_inds) + np.sum(anchor_labels == 0) == cfg.ANCHOR.BATCH_PER_IM
        return anchor_labels, anchor_boxes



    def reset_anchors(self,max_size=(512,512)):
        '''

        :param max_size: h,w
        :return:
        '''
        self.max_size=max_size

        self.anchors = self.produce_anchors()

if __name__=='__main__':
    ##model_eval the  anchor codes there
    import cv2

    cell_anchor = CellAnchor.generate_cell_anchor()
    print(cell_anchor)


    anchor_maker=Anchor()

    all_anchor= anchor_maker.anchors
    print(len(all_anchor))
    image=np.ones(shape=[cfg.DATA.max_size[0],cfg.DATA.max_size[1],3])*255

    # for x in anchors:
    #     print(x.shape)

    anchors=np.array(all_anchor)
    cv2.namedWindow('anchors', 0)
    for i in range(0,anchors.shape[0]):
        box=anchors[i]
        print(box[2]-box[0])
        cv2.rectangle(image, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (255, 0, 0), 1)


        cv2.imshow('anchors',image)
        cv2.waitKey(0)

    #a,b=anchor_maker.produce_target(image,np.array([[34., 396.,  58., 508.],[20,140,50,160]]),np.array([1,1]))




