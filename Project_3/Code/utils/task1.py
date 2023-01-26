import numpy as np
from shapely import geometry


def rotation_y(ry):

    cos = np.cos(ry)
    sin = np.sin(ry)

    rot = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])

    return rot


def label2corners(label):
    """
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
            6 -------- 7
           /|         /|
          5 -------- 4 .
          | |        | |
          . 2 -------- 3
          |/         |/
          1 -------- 0
    """
    corners = np.zeros((label.shape[0], 8, 3))

    for i in range(label.shape[0]):

        h = label[i, 3]
        l = label[i, 4]
        w = label[i, 5]
        ry = label[i, 6]

        # obtain from the three dimensions of the car the 8 points for the bb
        x_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2], dtype=np.float32).T
        y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h], dtype=np.float32).T
        z_corners = np.array([-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.], dtype=np.float32).T

        # rotate the 8 points with respect to the y axis of the center
        rot_mat = rotation_y(ry)
        get_corners = np.vstack([x_corners, y_corners, z_corners])
        rotated_corners = np.matmul(rot_mat, get_corners)

        x_center = label[i, 0]
        y_center = label[i, 1]
        z_center = label[i, 2]

        # add the coordinates of the center of the bb to the 8 corners
        rotated_corners[0, :] += x_center
        rotated_corners[1, :] += y_center
        rotated_corners[2, :] += z_center

        corners[i, :, :] = np.transpose(rotated_corners)

    return corners


def get_iou(pred, target):
    """
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    """
    old_pred = pred
    old_target = target
    pred = label2corners(pred)
    target = label2corners(target)

    iou_matrix = np.zeros((len(pred), len(target)))

    for i, current_pred in enumerate(pred):
        h_p = old_pred[i, 3]
        l_p = old_pred[i, 4]
        w_p = old_pred[i, 5]

        # create the 2D polygon representing the lower face of the predicted bb
        polygon_pred = geometry.Polygon(current_pred[0:4, (0, 2)])
        for j, current_target in enumerate(target):
            h_t = old_target[j, 3]
            l_t = old_target[j, 4]
            w_t = old_target[j, 5]

            y_min = min(current_pred[0, 1], current_target[0, 1])
            y_max = max(current_pred[4, 1], current_target[4, 1])

            # create the 2D polygon representing the lower face of the target bb
            polygon_target = geometry.Polygon(current_target[0:4, (0, 2)])

            # calculate the 2D intersection area of the 2 aforementioned polygons

            polygon_intersection = polygon_pred.intersection(polygon_target).area

            # calculate the volume of both the pred bb and the target bb
            p_vol = h_p*l_p*w_p
            t_vol = h_t*l_t*w_t

            # calculate the 3D intersection volume by multiplying the 2D intersection by the 1D intersection along y
            intersect3d = polygon_intersection * max(0.0, y_min - y_max)

            # calculate the iou through the known formula
            iou_matrix[i, j] = intersect3d/(p_vol + t_vol - intersect3d)
    return iou_matrix


def compute_recall(pred, target, threshold):
    """
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    """
    iou = get_iou(pred, target)
    iou_max = np.max(iou, 0)

    # count the number of targets which have at least 1 pred with iou > threshold (TP)
    boolean_tp = iou_max >= threshold
    tp = np.count_nonzero(boolean_tp)

    # count the number of targets which have no pred with iou > threshold (FN)
    boolean_fn = iou_max < threshold
    fn = np.count_nonzero(boolean_fn)

    recall = tp/(tp+fn)

    return recall


