import os
import numpy as np
import cv2
import random
import math
######May wrong, when use it check it
def Rotate_aug(src,angle,label=None,center=None,scale=1.0):
    '''
    :param src: src image
    :param label: label should be numpy array with [[x1,y1],
                                                    [x2,y2],
                                                    [x3,y3]...]
    :param angle:
    :param center:
    :param scale:
    :return: the rotated image and the points
    '''
    image=src
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if label is None:
        for i in range(image.shape[2]):
            image[:,:,i] = cv2.warpAffine(image[:,:,i], M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        return image,None
    else:
        label=label.T
        ####make it as a 3x3 RT matrix
        full_M=np.row_stack((M,np.asarray([0,0,1])))
        img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        ###make the label as 3xN matrix
        full_label = np.row_stack((label, np.ones(shape=(1,label.shape[1]))))
        label_rotated=np.dot(full_M,full_label)
        label_rotated=label_rotated[0:2,:]
        #label_rotated = label_rotated.astype(np.int32)
        label_rotated=label_rotated.T
        return img_rotated,label_rotated
def Rotate_coordinate(label,rt_matrix):
    if rt_matrix.shape[0]==2:
        rt_matrix=np.row_stack((rt_matrix, np.asarray([0, 0, 1])))
    full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
    label_rotated = np.dot(rt_matrix, full_label)
    label_rotated = label_rotated[0:2, :]
    return label_rotated


def box_to_point(boxes):
    '''

    :param boxes: [n,x,y,x,y]
    :return: [4n,x,y]
    '''
    ##caution the boxes are ymin xmin ymax xmax
    points_set=np.zeros(shape=[4*boxes.shape[0],2])

    for i in range(boxes.shape[0]):
        points_set[4 * i]=np.array([boxes[i][0],boxes[i][1]])
        points_set[4 * i+1] =np.array([boxes[i][0],boxes[i][3]])
        points_set[4 * i+2] =np.array([boxes[i][2],boxes[i][3]])
        points_set[4 * i+3] =np.array([boxes[i][2],boxes[i][1]])


    return points_set


def point_to_box(points):
    boxes=[]
    points=points.reshape([-1,4,2])

    for i in range(points.shape[0]):
        box=[np.min(points[i][:,0]),np.min(points[i][:,1]),np.max(points[i][:,0]),np.max(points[i][:,1])]

        boxes.append(box)

    return np.array(boxes)


def Rotate_with_box(src,angle,boxes=None,center=None,scale=1.0):
    '''
    :param src: src image
    :param label: label should be numpy array with [[x1,y1],
                                                    [x2,y2],
                                                    [x3,y3]...]
    :param angle:angel
    :param center:
    :param scale:
    :return: the rotated image and the points
    '''

    label=box_to_point(boxes)
    image=src
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心


    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)

    new_size=Rotate_coordinate(np.array([[0,w,w,0],
                                         [0,0,h,h]]), M)

    new_h,new_w=np.max(new_size[1])-np.min(new_size[1]),np.max(new_size[0])-np.min(new_size[0])

    scale=min(h/new_h,w/new_w)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    if boxes is None:
        for i in range(image.shape[2]):
            image[:,:,i] = cv2.warpAffine(image[:,:,i], M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        return image,None
    else:
        label=label.T
        ####make it as a 3x3 RT matrix
        full_M=np.row_stack((M,np.asarray([0,0,1])))
        img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        ###make the label as 3xN matrix
        full_label = np.row_stack((label, np.ones(shape=(1,label.shape[1]))))
        label_rotated=np.dot(full_M,full_label)
        label_rotated=label_rotated[0:2,:]
        #label_rotated = label_rotated.astype(np.int32)
        label_rotated=label_rotated.T

        boxes_rotated = point_to_box(label_rotated)
        return img_rotated,boxes_rotated

###CAUTION:its not ok for transform with label for perspective _aug
def Perspective_aug(src,strength,label=None):
    image = src
    pts_base = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts1=np.random.rand(4, 2)*random.uniform(-strength,strength)+pts_base
    pts1=pts1.astype(np.float32)
    #pts1 =np.float32([[56, 65], [368, 52], [28, 387], [389, 398]])
    M = cv2.getPerspectiveTransform(pts1, pts_base)
    trans_img = cv2.warpPerspective(image, M, (src.shape[1], src.shape[0]))

    label_rotated=None
    if label is not  None:
        label=label.T
        full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
        label_rotated = np.dot(M, full_label)
        label_rotated=label_rotated.astype(np.int32)
        label_rotated=label_rotated.T
    return trans_img,label_rotated


def Affine_aug(src,strength,label=None):
    image = src
    pts_base = np.float32([[10,100],[200,50],[100,250]])
    pts1 = np.random.rand(3, 2) * random.uniform(-strength, strength) + pts_base
    pts1 = pts1.astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts_base)
    trans_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    label_rotated=None
    if label is not None:
        label=label.T
        full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
        label_rotated = np.dot(M, full_label)
        #label_rotated = label_rotated.astype(np.int32)
        label_rotated=label_rotated.T
    return trans_img,label_rotated
def Padding_aug(src,max_pattern_ratio=0.05):

    pattern=np.ones_like(src)
    ratio = random.uniform(0, max_pattern_ratio)
    width=src.shape[1]
    height=src.shape[0]
    if random.uniform(0,1)>0.5:
        if random.uniform(0, 1) > 0.5:
            pattern[0:int(ratio*height),:,:]=0
        else:
            pattern[-int(ratio * height):-1, :, :] = 0
    else:
        if random.uniform(0, 1) > 0.5:
            pattern[:,0:int(ratio * width), :] = 0
        else:
            pattern[:,-int(ratio * width):-1,  :] = 0
    img=src*pattern
    return img


def Fill_img(img_raw,target_height,target_width,label=None):

    ###sometimes use in objs detects
    channel=img_raw.shape[2]
    raw_height = img_raw.shape[0]
    raw_width = img_raw.shape[1]
    if raw_width / raw_height >= target_width / target_height:
        shape_need = [int(target_height / target_width * raw_width), raw_width, channel]
        img_fill = np.zeros(shape_need, dtype=img_raw.dtype)+np.array(cfg.DATA.PIXEL_MEAN,dtype=img_raw.dtype)
        shift_x=(img_fill.shape[1]-raw_width)//2
        shift_y=(img_fill.shape[0]-raw_height)//2
        for i in range(channel):
            img_fill[shift_y:raw_height+shift_y, shift_x:raw_width+shift_x, i] = img_raw[:,:,i]
    else:
        shape_need = [raw_height, int(target_width / target_height * raw_height), channel]
        img_fill = np.zeros(shape_need, dtype=img_raw.dtype)+np.array(cfg.DATA.PIXEL_MEAN,dtype=img_raw.dtype)
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
def Random_crop(src,shrink):
    h,w,_=src.shape

    h_shrink=int(h*shrink)
    w_shrink = int(w * shrink)
    bimg = cv2.copyMakeBorder(src, h_shrink, h_shrink, w_shrink, w_shrink, borderType=cv2.BORDER_CONSTANT,
                              value=(0,0,0))

    start_h=random.randint(0,2*h_shrink)
    start_w=random.randint(0,2*w_shrink)

    target_img=bimg[start_h:start_h+h,start_w:start_w+w,:]

    return target_img

def box_in_img(img,boxes,min_overlap=0.5):

    raw_bboxes = np.array(boxes)

    face_area=(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])

    h,w,_=img.shape
    boxes[:, 0][boxes[:, 0] <=0] =0
    boxes[:, 0][boxes[:, 0] >=w] = w
    boxes[:, 2][boxes[:, 2] <= 0] = 0
    boxes[:, 2][boxes[:, 2] >= w] = w

    boxes[:, 1][boxes[:, 1] <= 0] = 0
    boxes[:, 1][boxes[:, 1] >= h] = h

    boxes[:, 3][boxes[:, 3] <= 0] = 0
    boxes[:, 3][boxes[:, 3] >= h] = h

    boxes_in = []
    for i in range(boxes.shape[0]):
        box=boxes[i]
        if ((box[3]-box[1])*(box[2]-box[0]))/face_area[i]>min_overlap :
            boxes_in.append(boxes[i])

    boxes_in = np.array(boxes_in)
    return boxes_in


def Random_scale_withbbox(image,bboxes,target_shape,jitter=0.5):

    ###the boxes is in ymin,xmin,ymax,xmax mode
    hi, wi, _ = image.shape

    while 1:
        if len(bboxes)==0:
            print('errrrrrr')
        bboxes_=np.array(bboxes)
        crop_h = int(hi * random.uniform(0.2, 1))
        crop_w = int(wi * random.uniform(0.2, 1))

        start_h = random.randint(0, hi - crop_h)
        start_w = random.randint(0, wi - crop_w)

        croped = image[start_h:start_h + crop_h, start_w:start_w + crop_w, :]

        bboxes_[:, 0] = bboxes_[:, 0] - start_w
        bboxes_[:, 1] = bboxes_[:, 1] - start_h
        bboxes_[:, 2] = bboxes_[:, 2] - start_w
        bboxes_[:, 3] = bboxes_[:, 3] - start_h


        bboxes_fix=box_in_img(croped,bboxes_)
        if len(bboxes_fix)>0:
            break


    ###use box
    h,w=target_shape
    croped_h,croped_w,_=croped.shape

    croped_h_w_ratio=croped_h/croped_w

    rescale_h=int(h * random.uniform(0.5, 1))

    rescale_w = int(rescale_h/(random.uniform(0.7, 1.3)*croped_h_w_ratio))
    rescale_w=np.clip(rescale_w,0,w)

    image=cv2.resize(croped,(rescale_w,rescale_h))

    bboxes_fix[:, 0] = bboxes_fix[:, 0] * rescale_w/ croped_w
    bboxes_fix[:, 1] = bboxes_fix[:, 1] * rescale_h / croped_h
    bboxes_fix[:, 2] = bboxes_fix[:, 2] * rescale_w / croped_w
    bboxes_fix[:, 3] = bboxes_fix[:, 3] * rescale_h / croped_h



    return image,bboxes_fix


def Random_flip(im, boxes):

    im_lr = np.fliplr(im).copy()
    h,w,_ = im.shape
    xmin = w - boxes[:,2]
    xmax = w - boxes[:,0]
    boxes[:,0] = xmin
    boxes[:,2] = xmax
    return im_lr, boxes










def Mirror(src,label=None,symmetry=None):

    img = cv2.flip(src, 1)
    if label is None:
        return img

    width=img.shape[1]
    cod = []
    allc = []
    for i in range(label.shape[0]):
        x, y = label[i][0], label[i][1]
        if x >= 0:
            x = width - 1 - x
        cod.append((x, y))
    # **** the joint index depends on the dataset ****
    for (q, w) in symmetry:
        cod[q], cod[w] = cod[w], cod[q]
    for i in range(label.shape[0]):
        allc.append(cod[i][0])
        allc.append(cod[i][1])
    label = np.array(allc).reshape(label.shape[0], 2)
    return img,label


class RandomBaiduCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self, size):

        self.mean = np.array([103, 116, 123], dtype=np.float32)
        self.maxSize = 12000  # max size
        self.infDistance = 9999999
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        random_counter = 0
        boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        # argsort = np.argsort(boxArea)
        # rand_idx = random.randint(min(len(argsort),6))
        # print('rand idx',rand_idx)
        rand_idx = random.randint(0,len(boxArea)-1)
        rand_Side = boxArea[rand_idx] ** 0.5
        # rand_Side = min(boxes[rand_idx,2] - boxes[rand_idx,0] + 1, boxes[rand_idx,3] - boxes[rand_idx,1] + 1)
        anchors = [16, 32, 64, 128, 256, 512]
        distance = self.infDistance
        anchor_idx = 5
        for i, anchor in enumerate(anchors):
            if abs(anchor - rand_Side) < distance:
                distance = abs(anchor - rand_Side)
                anchor_idx = i
        target_anchor = random.choice(anchors[0:min(anchor_idx + 1, 5) + 1])
        ratio = float(target_anchor) / rand_Side
        ratio = ratio * (2 ** random.uniform(-1, 1))
        if int(height * ratio * width * ratio) > self.maxSize * self.maxSize:
            ratio = (self.maxSize * self.maxSize / (height * width)) ** 0.5
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)
        image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)
        boxes[:, 0] *= ratio
        boxes[:, 1] *= ratio
        boxes[:, 2] *= ratio
        boxes[:, 3] *= ratio
        height, width, _ = image.shape
        sample_boxes = []
        xmin = boxes[rand_idx, 0]
        ymin = boxes[rand_idx, 1]
        bw = (boxes[rand_idx, 2] - boxes[rand_idx, 0] + 1)
        bh = (boxes[rand_idx, 3] - boxes[rand_idx, 1] + 1)
        w = h = self.size

        for _ in range(50):
            if w < max(height, width):
                if bw <= w:
                    w_off = random.uniform(xmin + bw - w, xmin)
                else:
                    w_off = random.uniform(xmin, xmin + bw - w)
                if bh <= h:
                    h_off = random.uniform(ymin + bh - h, ymin)
                else:
                    h_off = random.uniform(ymin, ymin + bh - h)
            else:
                w_off = random.uniform(width - w, 0)
                h_off = random.uniform(height - h, 0)
            w_off = math.floor(w_off)
            h_off = math.floor(h_off)
            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(w_off), int(h_off), int(w_off + w), int(h_off + h)])
            # keep overlap with gt box IF center in sampled patch
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            # mask in all gt boxes that above and to the left of centers
            m1 = (rect[0] <= boxes[:, 0]) * (rect[1] <= boxes[:, 1])
            # mask in all gt boxes that under and to the right of centers
            m2 = (rect[2] >= boxes[:, 2]) * (rect[3] >= boxes[:, 3])
            # mask in that both m1 and m2 are true
            mask = m1 * m2
            overlap = self.jaccard_numpy(boxes, rect)
            # have any valid boxes? try again if not
            if not mask.any() and not overlap.max() > 0.7:
                continue
            else:
                sample_boxes.append(rect)

        if len(sample_boxes) > 0:
            choice_idx = random.randint(0,len(sample_boxes)-1)
            choice_box = sample_boxes[choice_idx]
            # print('crop the box :',choice_box)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            m1 = (choice_box[0] < centers[:, 0]) * (choice_box[1] < centers[:, 1])
            m2 = (choice_box[2] > centers[:, 0]) * (choice_box[3] > centers[:, 1])
            mask = m1 * m2
            current_boxes = boxes[mask, :].copy()
            current_labels = labels[mask]
            current_boxes[:, :2] -= choice_box[:2]
            current_boxes[:, 2:] -= choice_box[:2]
            if choice_box[0] < 0 or choice_box[1] < 0:
                new_img_width = width if choice_box[0] >= 0 else width - choice_box[0]
                new_img_height = height if choice_box[1] >= 0 else height - choice_box[1]
                image_pad = np.zeros((new_img_height, new_img_width, 3), dtype=float)+np.array(cfg.DATA.PIXEL_MEAN,dtype=float)
                start_left = 0 if choice_box[0] >= 0 else -choice_box[0]
                start_top = 0 if choice_box[1] >= 0 else -choice_box[1]
                image_pad[start_top:, start_left:, :] = image

                choice_box_w = choice_box[2] - choice_box[0]
                choice_box_h = choice_box[3] - choice_box[1]

                start_left = choice_box[0] if choice_box[0] >= 0 else 0
                start_top = choice_box[1] if choice_box[1] >= 0 else 0
                end_right = start_left + choice_box_w
                end_bottom = start_top + choice_box_h
                current_image = image_pad[start_top:end_bottom, start_left:end_right, :].copy()
                return current_image, current_boxes, current_labels
            current_image = image[choice_box[1]:choice_box[3], choice_box[0]:choice_box[2], :].copy()
            return current_image, current_boxes, current_labels
        else:
            return image, boxes, labels
    def jaccard_numpy(self,box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.
        Args:
            box_a: Multiple bounding boxes, Shape: [num_boxes,4]
            box_b: Single bounding box, Shape: [4]
        Return:
            jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
        """
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                  (box_a[:, 3]-box_a[:, 1]))  # [A,B]
        area_b = ((box_b[2]-box_b[0]) *
                  (box_b[3]-box_b[1]))  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]


    def intersect(self,box_a, box_b):
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]

import sys
sys.path.append('.')
from train_config import config as cfg
baidu_aug=RandomBaiduCrop(cfg.DATA.hin)

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(0,width - w)
                top = random.uniform(0,height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = self.jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels

    def jaccard_numpy(self,box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.
        Args:
            box_a: Multiple bounding boxes, Shape: [num_boxes,4]
            box_b: Single bounding box, Shape: [4]
        Return:
            jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
        """
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                  (box_a[:, 3]-box_a[:, 1]))  # [A,B]
        area_b = ((box_b[2]-box_b[0]) *
                  (box_b[3]-box_b[1]))  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]


    def intersect(self,box_a, box_b):
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]
dsfd_aug=RandomSampleCrop()

if __name__=='__main__':


    import sys
    sys.path.append('.')
    from train_config import config as cfg


    from lib.dataset.augmentor.visual_augmentation import ColorDistort
    color_aug=ColorDistort()
    for i in range(1000):
        image=cv2.imread('./lib/dataset/augmentor/model_eval.jpg')
        boxes = np.array([[165, 60, 233, 138,1]],dtype=np.float)
        #bboxes=np.array([[165,60,233,138],[100,60,233,138]])

        if random.uniform(0, 1) > 0.5:
            image, boxes = Random_flip(image, boxes)
        if random.uniform(0, 1) > 0.5:
            image=color_aug(image)
        # if random.uniform(0, 1) > 0.7:
        #     boxes_ = boxes[:, 0:4]
        #     klass_ = boxes[:, 4:]
        #     angle = random.sample([-90, 90], 1)[0]
        #     image, boxes_ = Rotate_with_box(image, boxes=boxes_, angle=angle)
        #     boxes = np.concatenate([boxes_, klass_], axis=1)

        sample_dice=random.uniform(0, 1)
        if  sample_dice> 0.7:
            if not cfg.DATA.MUTISCALE:
                image, boxes = Random_scale_withbbox(image, boxes, target_shape=[cfg.DATA.hin, cfg.DATA.win],
                                                     jitter=0.3)
            else:
                rand_h = random.sample(cfg.DATA.scales, 1)[0]
                rand_w = random.sample(cfg.DATA.scales, 1)[0]
                image, boxes = Random_scale_withbbox(image, boxes, target_shape=[rand_h, rand_w], jitter=0.3)
        elif sample_dice>0.3 and sample_dice<=0.7:
            boxes_ = boxes[:, 0:4]
            klass_ = boxes[:, 4:]

            image, boxes_, klass_ = dsfd_aug(image, boxes_, klass_)
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
        else:
            boxes_ = boxes[:, 0:4]
            klass_ = boxes[:, 4:]
            image,boxes_,klass_=baidu_aug(image,boxes_,klass_)

            image, shift_x, shift_y = Fill_img(image, target_width=cfg.DATA.win, target_height=cfg.DATA.hin)
            boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
            h, w, _ = image.shape
            boxes_[:, 0] /= w
            boxes_[:, 1] /= h
            boxes_[:, 2] /= w
            boxes_[:, 3] /= h
            image=image.astype(np.uint8)
            image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin))

            boxes_[:, 0] *= cfg.DATA.win
            boxes_[:, 1] *= cfg.DATA.hin
            boxes_[:, 2] *= cfg.DATA.win
            boxes_[:, 3] *= cfg.DATA.hin
            boxes = np.concatenate([boxes_, klass_], axis=1)

        if np.sum(image)==0:
            print('there is an err with 2')

        for i in range(boxes.shape[0]):
            box = boxes[i]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (222, 222, 100), 1)

        cv2.imshow('tmp1', image)
        cv2.waitKey(0)


