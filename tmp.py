
import numpy as np

def py_nms(bboxes, iou_thres,score_thres):



    upper_thres = np.where(bboxes[:,4] > score_thres)[0]

    bboxes=bboxes[upper_thres]



    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    order = np.argsort(bboxes[:, 4])[::-1]


    keep = []

    while order.shape[0] > 0:
        cur = order[0]

        keep.append(cur)

        area = (bboxes[cur, 2] - bboxes[cur, 0]) * (bboxes[cur, 3] - bboxes[cur, 1])

        x1_reain = x1[order[1:]]
        y1_reain = y1[order[1:]]
        x2_reain = x2[order[1:]]
        y2_reain = y2[order[1:]]


        xx1 = np.maximum(bboxes[cur, 0], x1_reain)
        yy1 = np.maximum(bboxes[cur, 1], y1_reain)
        xx2 = np.minimum(bboxes[cur, 2], x2_reain)
        yy2 = np.minimum(bboxes[cur, 3], y2_reain)

        intersection = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)

        iou = intersection / (area + (y2_reain - y1_reain) * (x2_reain - x1_reain) - intersection)

        ##keep the low iou
        low_iou_position = np.where(iou < iou_thres)[0]

        order = order[low_iou_position + 1]


    return bboxes[keep]





if __name__=='__main__':



    box=np.array([[1,1,20,20,0.9],
                  [3,3,19,19,0.8],
                  [6,7,18,18,0.03],
                  [50,60,70,80,0.5]])


    res=py_nms(box,0.4,0.6)


