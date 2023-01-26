import numpy as np
from .task1 import label2corners


def duplicate_vector(vector, out_size):
    if vector.ndim > 1:
        vector = vector.ravel()

    final_vector = vector
    rand_indices = np.random.choice(vector, (out_size - np.shape(final_vector)[0], 1), replace=True)
    final_vector = np.append(final_vector, rand_indices)
    final_vector = np.expand_dims(np.array(final_vector, dtype=int), 1)

    return final_vector


def dir_sizes(cubes3d):
    b1 = cubes3d[:, 1, :]
    b2 = cubes3d[:, 2, :]
    b0 = cubes3d[:, 0, :]
    t5 = cubes3d[:, 5, :]

    # direction 0-1 (x)
    dir1 = (b0 - b1)
    size1 = np.linalg.norm(dir1, axis=1)
    size1 = np.expand_dims(size1, axis=1)
    dir1 = dir1 / size1[:]

    # direction 5-1 (y)
    dir2 = (t5 - b1)
    size2 = np.linalg.norm(dir2, axis=1)
    size2 = np.expand_dims(size2, axis=1)
    dir2 = dir2 / size2[:]

    # direction 2-1 (z)
    dir3 = (b2 - b1)
    size3 = np.linalg.norm(dir3, axis=1)
    size3 = np.expand_dims(size3, axis=1)
    dir3 = dir3 / size3[:]

    directions = np.stack((dir1, dir2, dir3), axis=1)

    sizes = np.stack((size1, size2, size3), axis=1)

    return directions, sizes

def inside_test(points, cube3d, dir, size):
    """
    cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
    points = array of points with shape (N, 3).
    Returns the indices of the points array which are inside the cube3d
            6 -------- 7
           /|         /|
          5 -------- 4 .
          | |        | |
          . 2 -------- 3
          |/         |/
          1 -------- 0
    """

    b0 = cube3d[0, :]
    t6 = cube3d[6, :]
    # calculate the coordinates of the center of the bb
    cube3d_center = (b0 + t6) / 2.0

    # vector which goes from the center of the bb to the point taken into account
    dir_vec = np.subtract(points, cube3d_center, dtype=np.float32)

    proj_vec = np.absolute(np.matmul(dir_vec, dir, dtype=np.float32)) * 2

    # project the dir_vec along the 3 directions and check if these are all smaller than the size of the 3 axes of the bb
    inside_points = np.where((proj_vec[:, 0] <= size[0]) & (proj_vec[:, 1] <= size[1]) & (proj_vec[:, 2] <= size[2]))[0]
    return inside_points


def selected_indices(inside_ind, max_p):
    """
    Creates the array of features of the points when we have less than 512 points
    if np.shape(points)[0] != max_p:
        rand_ind = np.random.randint(0, np.shape(points)[0], max_p)
        points = points[rand_ind, :]
        feat = feat[rand_ind, :]
    """

    if np.shape(inside_ind)[0] >= max_p:
        rand_ind = np.random.choice(inside_ind, (max_p, 1), replace=False)

    elif np.shape(inside_ind)[0] < max_p:
        rand_ind = duplicate_vector(inside_ind, max_p)

    return rand_ind


def add_delta(label, delta):
    # Add 1 meter along each direction to increase the bb
    problem_2 = True

    label[:, (3, 4, 5)] += 2 * delta

    if not problem_2:
        label[:, 1] += delta

    return label


def roi_pool(pred, xyz, feat, config):
    """
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie
       in each enlarged bounding box. Each ROI should contain exactly 512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (N,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K',7)
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    """

    max_points = config['max_points']
    delta = config['delta']

    valid_pred = []
    indices = []

    pred_corn = label2corners(add_delta(pred.copy(), delta))

    x_max = np.amax(pred_corn[:, :, 0])
    x_min = np.amin(pred_corn[:, :, 0])
    y_max = np.amax(pred_corn[:, :, 1])
    y_min = np.amin(pred_corn[:, :, 1])
    z_max = np.amax(pred_corn[:, :, 2])
    z_min = np.amin(pred_corn[:, :, 2])

    xyz_indices = np.where((xyz[:, 0] <= x_max) & (xyz[:, 1] <= y_max) & (xyz[:, 2] <= z_max) &
                           (xyz[:, 0] >= x_min) & (xyz[:, 1] >= y_min) & (xyz[:, 2] >= z_min))[0]

    xyz = xyz[xyz_indices, :]
    feat = feat[xyz_indices, :]

    directions, sizes = dir_sizes(pred_corn)

    for i, box in enumerate(pred_corn):
        # obtain the indices of the points inside the specific bb under consideration
        in_ind = inside_test(xyz, box, directions[i, :, :].T, sizes[i, :, :])

        if len(in_ind) != 0:

            # if there is at least 1 point inside the bb, select the valid pred
            valid_pred.append(pred[i, :])

            # sample exactly 512 points
            sel_ind = selected_indices(in_ind, max_points)

            indices.append(sel_ind)

    indices = np.array(indices).squeeze()
    pooled_xyz = xyz[indices, :]
    pooled_feat = feat[indices, :]

    valid_pred = np.array(valid_pred)
    pooled_xyz = np.array(pooled_xyz)
    pooled_feat = np.array(pooled_feat)

    return valid_pred, pooled_xyz, pooled_feat

