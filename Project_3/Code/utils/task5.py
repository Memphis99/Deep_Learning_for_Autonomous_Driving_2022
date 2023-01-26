import numpy as np

from .task1 import get_iou


def nms(pred, score, threshold):
    """
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresponding confidence scores
    """
    # threshold = 0.01

    s_f = np.empty((0, 7))
    c_f = np.empty((0, 1))
    new_pred = pred.copy()

    # needed in order to obtain BEV iou
    new_pred[:, 1] = 0
    new_pred[:, 3] = 1

    while np.shape(pred)[0] != 0:
        c = np.argmax(score)
        d = np.reshape(new_pred[c, :], (-1, np.shape(pred)[1]))
        d_insert = np.reshape(pred[c, :], (-1, np.shape(pred)[1]))

        pred = np.delete(pred, c, 0)
        new_pred = np.delete(new_pred, c, 0)

        s_f = np.append(s_f, d_insert.reshape(-1, 7), axis=0)
        c_f = np.append(c_f, score[c].reshape(-1, 1), axis=0)

        score = np.delete(score, c, 0)

        iou = get_iou(new_pred, d)
        iou_ind_thr = np.where(iou >= threshold)

        new_pred = np.delete(new_pred, iou_ind_thr, 0)
        pred = np.delete(pred, iou_ind_thr, 0)
        score = np.delete(score, iou_ind_thr, 0)

    return s_f, c_f