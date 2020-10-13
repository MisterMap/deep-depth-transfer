import torch.nn as nn

from deep_depth_transfer.data.cameras_calibration import CamerasCalibration
from .inverse_depth_smoothness_loss import InverseDepthSmoothnessLoss
from .temporal_photometric_consistency_loss import TemporalPhotometricConsistencyLoss


class MonoUnsupervisedCriterion(nn.Module):
    def __init__(self,
                 cameras_calibration: CamerasCalibration,
                 lambda_s=0.85,
                 lambda_smoothness=1.0):
        super(MonoUnsupervisedCriterion, self).__init__()

        self._temporal_consistency_loss = TemporalPhotometricConsistencyLoss(
            cameras_calibration.left_camera_matrix,
            cameras_calibration.right_camera_matrix,
            lambda_s,
        )
        self._inverse_depth_smoothness_loss = InverseDepthSmoothnessLoss(
            lambda_smoothness,
        )

    def forward(self, images, depths, transformations):
        current_image, next_image = images
        current_depth, next_depth = depths
        current_transform, next_transform = transformations

        smoothness_losses = [self._inverse_depth_smoothness_loss(x, y) for x, y in zip(depths, images)]

        temporal_loss = self._temporal_consistency_loss(
            current_image,
            next_image,
            current_depth,
            next_depth,
            current_transform[1],
            current_transform[0],
            next_transform[1],
            next_transform[0]
        )

        losses = {
            "temporal_loss": temporal_loss,
            "smooth_loss": sum(smoothness_losses) / len(smoothness_losses),
            "loss": temporal_loss + sum(smoothness_losses) / len(smoothness_losses)
        }
        return losses
