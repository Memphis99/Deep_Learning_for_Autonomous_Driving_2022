import numpy as np

from .task1 import get_iou
from .task2 import duplicate_vector


def background_proposal(bg_e_indices, bg_h_indices, bg_indices, dim):
    # both hard and easy bg
    if (len(bg_e_indices) != 0) & (len(bg_h_indices) != 0):
        # hard samples
        if np.size(bg_h_indices) >= np.int(np.floor(dim / 2)):
            hard_indices = np.random.choice(bg_h_indices, (np.int(np.floor(dim / 2)), 1), replace=False)
        else:
            hard_indices = duplicate_vector(bg_h_indices, np.int(np.floor(dim / 2)))

        # easy samples
        if np.size(bg_e_indices) >= np.int(np.ceil(dim / 2)):
            easy_indices = np.random.choice(bg_e_indices, (np.int(np.ceil(dim / 2)), 1), replace=False)
        else:
            easy_indices = duplicate_vector(bg_e_indices, np.int(np.ceil(dim / 2)))

        out_ind = np.vstack((easy_indices, hard_indices))

    # just easy or hard bg
    else:
        if np.size(bg_indices) >= np.int(dim):
            out_ind = np.random.choice(bg_indices, (np.int(dim), 1), replace=False)
        else:
            out_ind = duplicate_vector(bg_indices, np.int(dim))

    return out_ind


def sample_proposals(pred, target, xyz, feat, config, train=False):
    """
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (string) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices
        feat (64,512,C) indices
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples)
    """

    t_bg_hard_lb = config['t_bg_hard_lb']
    t_bg_up = config['t_bg_up']
    t_fg_lb = config['t_fg_lb']

    n_samples = 64

    # task 3a
    iou_tab = get_iou(pred, target)
    selected_indices = np.argmax(iou_tab, 1)
    iou = np.max(iou_tab, 1)
    assigned_targets = target[selected_indices, :]

    # task 3b
    if train:

        # IOU threshold criterion
        fg_indices = np.where(iou > t_fg_lb)[0]
        bg_indices = np.where(iou < t_bg_up)[0]

        bg_e_indices = np.where(iou < t_bg_hard_lb)[0]
        bg_h_indices = np.where((t_bg_hard_lb < iou) & (iou < t_bg_up))[0]

        # max IOU criterion
        additional_fg = np.argmax(iou_tab, 0)

        new_additional_fg = np.delete(additional_fg, np.where(iou[additional_fg] > t_fg_lb))
        new_additional_fg = np.reshape(new_additional_fg, (np.shape(new_additional_fg)[0], 1))
        fg_indices = np.reshape(fg_indices, (np.shape(fg_indices)[0], 1))
        fg_indices = np.vstack((fg_indices, new_additional_fg))

        # target creation
        target = target[selected_indices, :]
        # both foreground and background
        if (len(fg_indices) != 0) & (len(bg_indices) != 0):
            # foreground samples (<=32)
            if np.shape(fg_indices)[0] >= int(n_samples/2):
                first_indices = np.random.choice(fg_indices[:, 0], (int(n_samples / 2), 1), replace=False)
            else:
                first_indices = fg_indices

            # background samples
            second_indices = background_proposal(bg_e_indices, bg_h_indices, bg_indices, n_samples - np.shape(first_indices)[0])

            indices = np.vstack((first_indices, second_indices))

        # just foreground or background
        else:
            # just background
            if np.size(bg_indices) != 0:
                indices = background_proposal(bg_e_indices, bg_h_indices, bg_indices, n_samples)

            # just foreground
            else:
                if np.size(fg_indices) >= np.int(n_samples):
                    indices = np.random.choice(fg_indices[:, 0], (np.int(n_samples), 1), replace=False)
                else:
                    indices = duplicate_vector(fg_indices, np.int(n_samples))

        iou = iou[indices]
        assigned_targets = target[indices, :].reshape((64, 7))

        xyz = xyz[indices, :, :].reshape((64, 512, 3))
        feat = feat[indices, :, :].reshape((64, 512, np.shape(feat)[2]))

    return assigned_targets, xyz, feat, iou
