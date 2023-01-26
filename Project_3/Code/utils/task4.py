import torch
import torch.nn as nn
import numpy as np


class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target, iou):
        """
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        """

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        positive_reg_lb = self.config['positive_reg_lb']
        positive_indices = np.where(iou.cpu() >= positive_reg_lb)[0]

        if len(positive_indices) == 0:
            loss = torch.zeros(1)
        else:
            pred_pos = pred[positive_indices, :].to(device)
            target_pos = target[positive_indices, :].to(device)

            loc_loss = self.loss(pred_pos[:, :3], target_pos[:, :3])
            size_loss = self.loss(pred_pos[:, 3:6], target_pos[:, 3:6])
            rotation_loss = self.loss(pred_pos[:, 6], target_pos[:, 6])

            loss = loc_loss + 3*size_loss + rotation_loss

        return loss.to(device)


class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        """
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        """

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        positive_cls_lb = self.config['positive_cls_lb']
        negative_cls_ub = self.config['negative_cls_ub']

        indices_zeros = np.where(iou.cpu() <= negative_cls_ub)[0]
        indices_ones = np.where(iou.cpu() >= positive_cls_lb)[0]
        indices = np.append(indices_zeros, indices_ones)
        indices = np.sort(indices)

        if len(indices) == 0:
            loss = torch.zeros(1)
        else:
            new_pred = pred[indices].to(device)
            target = torch.empty((np.shape(pred)[0], 1))
            target[indices_ones] = 1
            target[indices_zeros] = 0
            new_target = target[indices].to(device)

            loss = self.loss(new_pred, new_target)

        return loss.to(device)