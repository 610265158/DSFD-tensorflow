#-*-coding:utf-8-*-
import sys
sys.path.append('.')
import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse

from lib.core.api.face_detector import FaceDetector

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ap = argparse.ArgumentParser()
ap.add_argument( "--model", required=False, default='./model/detector.pb', help="model to eval:")
ap.add_argument( "--is_show", required=False, default=False, help="show result or not?")
ap.add_argument( "--data_dir", required=True, default="./FDDB/img", help="dir to img")
ap.add_argument( "--split_dir", required=True,default='./FDDB/FDDB-folds',help="dir to FDDB-folds")
ap.add_argument( "--result", required=True,default='./result',help="dir to write result")
args = ap.parse_args()


IMAGES_DIR = args.data_dir
ANNOTATIONS_PATH = args.split_dir
RESULT_DIR = args.result
MODEL_PATH = args.model

face_detector = FaceDetector([MODEL_PATH])


annotations = [s for s in os.listdir(ANNOTATIONS_PATH) if s.endswith('ellipseList.txt')]
image_lists = [s for s in os.listdir(ANNOTATIONS_PATH) if not s.endswith('ellipseList.txt')]
annotations = sorted(annotations)
image_lists = sorted(image_lists)

images_to_use = []
for n in image_lists:
    with open(os.path.join(ANNOTATIONS_PATH, n)) as f:
        images_to_use.extend(f.readlines())

images_to_use = [s.strip() for s in images_to_use]
with open(os.path.join(RESULT_DIR, 'faceList.txt'), 'w') as f:
    for p in images_to_use:
        f.write(p + '\n')


ellipses = []
for n in annotations:
    with open(os.path.join(ANNOTATIONS_PATH, n)) as f:
        ellipses.extend(f.readlines())

i = 0
with open(os.path.join(RESULT_DIR, 'ellipseList.txt'), 'w') as f:
    for p in ellipses:

        # check image order
        if 'big/img' in p:
            assert images_to_use[i] in p
            i += 1

        f.write(p)

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets

predictions = []
for n in tqdm(images_to_use):
    image_array = cv2.imread(os.path.join(IMAGES_DIR, n) + '.jpg')
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # threshold is important to set low


    boxes = face_detector(image_array, score_threshold=0.05)

    ##flip det
    flip_img=np.flip(image_array,1)

    boxes_flip_ = face_detector(flip_img, score_threshold=0.01)
    boxes_flip = np.zeros(boxes_flip_.shape)
    boxes_flip[:, 0] = flip_img.shape[1] - boxes_flip_[:, 2]
    boxes_flip[:, 1] = boxes_flip_[:, 1]
    boxes_flip[:, 2] = flip_img.shape[1] - boxes_flip_[:, 0]
    boxes_flip[:, 3] = boxes_flip_[:, 3]
    boxes_flip[:, 4] = boxes_flip_[:, 4]

    #####
    det = np.row_stack((boxes, boxes_flip))

    dets = bbox_vote(det)

    if args.is_show:
        for bbox in dets:
            if bbox[4] > 0.3:
                # cv2.circle(img_show,(p[0],p[1]),3,(0,0,213),-1)
                cv2.rectangle(image_array, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (255, 0, 0), 7)
        cv2.imshow('tmp', image_array)
        cv2.waitKey(0)


    ###


    predictions.append((n, dets[:,0:4], dets[:,4]))


with open(os.path.join(RESULT_DIR, 'detections.txt'), 'w') as f:
    for n, boxes, scores in predictions:
        f.write(n + '\n')
        f.write(str(len(boxes)) + '\n')
        for b, s in zip(boxes, scores):
            xmin, ymin, xmax, ymax = b
            h, w = int(ymax - ymin+1), int(xmax - xmin+1)
            f.write('{0} {1} {2} {3} {4:.4f}\n'.format(int(xmin), int(ymin), w, h, s))


