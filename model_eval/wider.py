import sys
sys.path.append('.')
import os
import scipy.io as sio
import argparse
import cv2
import numpy as np
import tensorflow as tf

import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from lib.core.api.face_detector import FaceDetector

ap = argparse.ArgumentParser()
ap.add_argument( "--model", required=False, default='./model/detector.pb', help="model to eval:")
ap.add_argument( "--is_show", required=False, default=False, help="show result or not?")
ap.add_argument( "--data_dir", required=False, default="./WIDER/WIDER_val", help="dir to img")
ap.add_argument( "--result", required=True,default='./result',help="dir to write result")
args = ap.parse_args()


IMAGES_DIR = args.data_dir
RESULT_DIR = args.result
MODEL_PATH = args.model

face_detector = FaceDetector([MODEL_PATH])
def get_data():
    subset = 'val'
    if subset is 'val':
        wider_face = sio.loadmat(
            './eval_tools/ground_truth/wider_face_val.mat')
    else:
        wider_face = sio.loadmat(
            './eval_tools/ground_truth/wider_face_test.mat')
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    imgs_path = os.path.join(IMAGES_DIR,'images')
    save_path = RESULT_DIR

    return event_list, file_list, imgs_path, save_path


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

def detect_face(img, shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink,
                         interpolation=cv2.INTER_LINEAR)

    detections = face_detector(img, score_threshold=0.05)

    det_xmin =  detections[:, 0] / shrink
    det_ymin =  detections[:, 1] / shrink
    det_xmax =  detections[:, 2] / shrink
    det_ymax =  detections[:, 3] / shrink
    det_conf =  detections[:, 4]
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))


    return det


def multi_scale_test( image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    index = np.where(np.maximum(
        det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (
        st + max_im_shrink) / 2
    det_b = detect_face( image, bt)

    # enlarge small image x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face( image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face( image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b

def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


event_list, file_list, imgs_path, save_path = get_data()



for index, event in enumerate(event_list):
        print(event)
        filelist = file_list[index][0]
        path = os.path.join(save_path, event[0][0])
        if not os.path.exists(path):
            os.makedirs(path)

        for num, file in enumerate(filelist):
            im_name = file[0][0]
            in_file = os.path.join(imgs_path, event[0][0], im_name[:] + '.jpg')

            image_array = cv2.imread(in_file)
            img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # max_im_shrink = (0x7fffffff / 200.0 / (img.shape[0] * img.shape[1])) ** 0.5
            max_im_shrink = np.sqrt(
                2000 * 2000 / (img.shape[0] * img.shape[1]))
            max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
            
            shrink = max_im_shrink if max_im_shrink < 1 else 1

            det0 = detect_face(img, shrink)

            det1 = flip_test( img, shrink)
            ##flip det

            [det2, det3] = multi_scale_test( img, max_im_shrink)
            #####
            det = np.row_stack((det0, det1, det2, det3))
            dets = bbox_vote(det)


            if args.is_show:
                for bbox in dets:
                    if bbox[4]>0.3:
                        # cv2.circle(img_show,(p[0],p[1]),3,(0,0,213),-1)
                        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                                      (int(bbox[2]), int(bbox[3])), (255, 0, 0), 7)
                cv2.imshow('tmp',img)
                cv2.waitKey(0)

            fout = open(os.path.join(save_path, event[0][0], im_name + '.txt'), 'w')
            fout.write('{:s}\n'.format(event[0][0] + '/' + im_name + '.jpg'))
            fout.write('{:d}\n'.format(dets.shape[0]))
            for i in range(dets.shape[0]):
                xmin = dets[i][0]
                ymin = dets[i][1]
                xmax = dets[i][2]
                ymax = dets[i][3]
                score = dets[i][4]
                fout.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                           format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))
fout.close()

