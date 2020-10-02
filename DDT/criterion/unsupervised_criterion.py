import torch.nn as nn

from DDT.data.cameras_calibration import CamerasCalibration
from DDT.utils import ResultDataPoint
from .inverse_depth_smoothness_loss import InverseDepthSmoothnessLoss
from .pose_loss import PoseLoss
from .pose_metric import PoseMetric
from .spatial_photometric_consistency_loss import SpatialPhotometricConsistencyLoss
from .temporal_photometric_consistency_loss import TemporalPhotometricConsistencyLoss


class UnsupervisedCriterion(nn.Module):
    def __init__(self,
                 cameras_calibration: CamerasCalibration,
                 lambda_position,
                 lambda_angle,
                 lambda_s=0.85,
                 lambda_smoothness=1.0):
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
        self._inverse_depth_smoothness_loss = InverseDepthSmoothnessLoss(
            lambda_smoothness,
        )
        self._pose_loss = PoseLoss(
            lambda_position,
            lambda_angle,
            cameras_calibration.transform_from_left_to_right
        )
        self._pose_metric = PoseMetric()

    def forward(self, images, depths, transformations):
        left_current_image, left_next_image, right_current_image, right_next_image = images
        left_current_depth, left_next_depth, right_current_depth, right_next_depth = depths
        left_current_transform, left_next_transform, right_current_transform, right_next_transform = transformations

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

        smoothness_losses = [self._inverse_depth_smoothness_loss(x) for x in depths]

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

        losses = {
            "spatial_loss": (current_spatial_loss + next_spatial_loss) / 2.,
            "temporal_loss": (left_temporal_loss + right_temporal_loss) / 2.,
            "smooth_loss": sum(smoothness_losses) / len(smoothness_losses),
            "poss_loss": (current_pose_loss + next_pose_loss) / 2.,
            "loss": ((current_spatial_loss + next_spatial_loss) / 2. +
                     (left_temporal_loss + right_temporal_loss) / 2. +
                     (current_pose_loss + next_pose_loss) / 2. +
                     sum(smoothness_losses) / len(smoothness_losses))
        }
        return losses

    def calculate_relative_pose_error(self, left_current_output: ResultDataPoint, right_current_output: ResultDataPoint,
                                      left_next_output: ResultDataPoint, right_next_output: ResultDataPoint,
                                      delta_position, delta_angle,
                                      inverse_delta_position, inverse_delta_angle):
        left_current_loss = self.pose_metric.calculate_relative_pose_error(left_current_output.translation,
                                                                           left_current_output.rotation,
                                                                           delta_position, delta_angle)
        right_current_loss = self.pose_metric.calculate_relative_pose_error(right_current_output.translation,
                                                                            right_current_output.rotation,
                                                                            delta_position, delta_angle)
        left_next_loss = self.pose_metric.calculate_relative_pose_error(left_next_output.translation,
                                                                        left_next_output.rotation,
                                                                        inverse_delta_position, inverse_delta_angle)
        right_next_loss = self.pose_metric.calculate_relative_pose_error(right_next_output.translation,
                                                                         right_next_output.rotation,
                                                                         inverse_delta_position, inverse_delta_angle)
        return (left_current_loss + right_current_loss + left_next_loss + right_next_loss) / 4
