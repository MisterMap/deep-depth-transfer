from .unsupervised_depth_model import UnsupervisedDepthModel
from ..utils.result_visualizer_utils import show_inner_spatial_loss
import torch.nn as nn


class MultiUnsupervisedDepthModel(UnsupervisedDepthModel):
    def __init__(self, params, pose_net, depth_net, criterion, result_visualizer=None, stereo=True, mono=True,
                 use_ground_truth_poses=False, inner_criterions=None):
        super().__init__(params, pose_net, depth_net, criterion, result_visualizer, stereo, mono,
                         use_ground_truth_poses)
        if inner_criterions is None:
            inner_criterions = {}
        self._inner_criterions = inner_criterions
        for module in depth_net.modules():
            if type(module) == nn.Conv2d:
                module.padding_mode = "reflect"

    def loss(self, batch):
        images = self.get_images(batch)
        depths = []
        inner_depth_results = []
        for image in images:
            depths.append(self.depth(image))
            inner_depth_results.append(self._depth_net.get_inner_result())

        transformations = self.get_transformations(images, batch)

        losses = self._criterion(images, depths, transformations)
        for level, criterion in self._inner_criterions.items():
            inner_depths = [x[:, :, ::2 ** (level + 1), ::2 ** (level + 1)] for x in depths]
            inner_images = [x[level] for x in inner_depth_results]
            inner_losses = criterion(inner_images, inner_depths, transformations)
            losses["loss"] += inner_losses["loss"]
            for key, value in inner_losses.items():
                losses[f"{key}_{level}"] = value
        return losses

    def show_figures(self, batch, batch_index):
        if self._result_visualizer is None:
            return
        if not self._result_visualizer.batch_index == batch_index:
            return
        images = self.get_images(batch)
        depths = []
        inner_depth_results = []
        for image in images:
            depths.append(self.depth(image)[0])
            inner_depth_results.append(self._depth_net.get_inner_result())
        images = [image[0] for image in images]
        figure = self._result_visualizer(images, depths)
        self.logger.log_figure("depth_reconstruction", figure, self.global_step)

        for level, criterion in self._inner_criterions.items():
            inner_depths = [x[None, :, ::2 ** (level + 1), ::2 ** (level + 1)] for x in depths]
            inner_images = [x[level] for x in inner_depth_results]
            figure = show_inner_spatial_loss(inner_images, inner_depths, criterion.get_cameras_calibration(),
                                             dpi=100)
            self.logger.log_figure(f"inner_loss_{level}", figure, self.global_step)
