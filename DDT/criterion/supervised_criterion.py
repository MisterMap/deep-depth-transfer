import torch.nn as nn

from DDT.data.cameras_calibration import CamerasCalibration
from DDT.utils import ResultDataPoint
from .losses import SpatialLosses, TemporalImageLosses


class SupervisedCriterion(nn.Module):
    def __init__(self, lambda_loss):
        super(SupervisedCriterion, self).__init__()
        self.lambda_loss = lambda_loss
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, output, target):
        return self.lambda_loss * self.mse_loss(output, target)
