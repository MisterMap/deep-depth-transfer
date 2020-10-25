import torch.nn as nn

from deep_depth_transfer.data.cameras_calibration import CamerasCalibration
from .inverse_depth_smoothness_loss import InverseDepthSmoothnessLoss
from .pose_loss import PoseLoss
from .pose_metric import PoseMetric
from .spatial_photometric_consistency_loss import SpatialPhotometricConsistencyLoss
from .temporal_photometric_consistency_loss import TemporalPhotometricConsistencyLoss


class UnsupervisedCriterion(nn.Module):
    def __init__(self,
                 cameras_calibration: CamerasCalibration,
                 lambda_position=0.01,
                 lambda_angle=0.1,
                 lambda_s=0.85,
                 lambda_smoothness=1.0,
                 smooth_loss=True,
                 pose_loss=True
                 ):
        super(UnsupervisedCriterion, self).__init__()

        self._spatial_consistency_loss = SpatialPhotometricConsistencyLoss(
            lambda_s,
            cameras_calibration.left_camera_matrix,
            cameras_calibration.right_camera_matrix,
            cameras_calibration.transform_from_left_to_right,
        )

        self._temporal_consistency_loss = TemporalPhotometricConsistencyLoss(
            cameras_calibration.left_camera_matrix,
            cameras_calibration.right_camera_matrix,
            lambda_s,
        )
        if smooth_loss:
            self._inverse_depth_smoothness_loss = InverseDepthSmoothnessLoss(
                lambda_smoothness,
            )
        else:
            self._inverse_depth_smoothness_loss = None
        if pose_loss:
            self._pose_loss = PoseLoss(
                lambda_position,
                lambda_angle,
                cameras_calibration.transform_from_left_to_right
            )
        else:
            self._pose_loss = None
        self._pose_metric = PoseMetric()
        self._cameras_calibration = cameras_calibration

    def get_cameras_calibration(self):
        return self._cameras_calibration

    def forward(self, images, depths, transformations):
        left_current_image, left_next_image, right_current_image, right_next_image = images
        left_current_depth, left_next_depth, right_current_depth, right_next_depth = depths
        left_current_transform, left_next_transform, right_current_transform, right_next_transform = transformations

        losses = {}
        current_spatial_loss = self._spatial_consistency_loss(
            left_current_image,
            right_current_image,
            left_current_depth,
            right_current_depth
        )

        next_spatial_loss = self._spatial_consistency_loss(
            left_next_image,
            right_next_image,
            left_next_depth,
            right_next_depth
        )

        losses["spatial_loss"] = (current_spatial_loss + next_spatial_loss) / 2.

        if self._inverse_depth_smoothness_loss is not None:
            smoothness_losses = [self._inverse_depth_smoothness_loss(x, y) for x, y in zip(depths, images)]
            losses["smooth_loss"] = sum(smoothness_losses) / len(smoothness_losses)

        left_temporal_loss = self._temporal_consistency_loss(
            left_current_image,
            left_next_image,
            left_current_depth,
            left_next_depth,
            left_current_transform[1],
            left_current_transform[0],
            left_next_transform[1],
            left_next_transform[0]
        )

        right_temporal_loss = self._temporal_consistency_loss(
            right_current_image,
            right_next_image,
            right_current_depth,
            right_next_depth,
            right_current_transform[1],
            right_current_transform[0],
            right_next_transform[1],
            right_next_transform[0]
        )

        losses["temporal_loss"] = (left_temporal_loss + right_temporal_loss) / 2.
        if self._pose_loss is not None:
            current_pose_loss = self._pose_loss(
                left_current_transform[1],
                left_current_transform[0],
                right_current_transform[1],
                right_current_transform[0],
            )

            next_pose_loss = self._pose_loss(
                left_next_transform[1],
                left_next_transform[0],
                right_next_transform[1],
                right_next_transform[0],
            )
            losses["poss_loss"] = (current_pose_loss + next_pose_loss) / 2.

        losses["loss"] = 0
        for value in losses.values():
            losses["loss"] += value
        return losses
