import torch.nn as nn

from .unsupervised_depth_model import UnsupervisedDepthModel
from ..utils.result_visualizer_utils import show_inner_spatial_loss


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
        self._down_samples = {}
        self._up_samples = {}
        print(self.hparams.depth_down_sample)
        for level in self._inner_criterions.keys():
            kernel_size = 2 ** (level + 1)
            if self.hparams.depth_down_sample == "avg":
                self._down_samples[level] = nn.AvgPool2d(kernel_size, kernel_size)
                self._up_samples[level] = nn.Identity()
            elif self.hparams.depth_down_sample == "min":
                self._down_samples[level] = nn.MaxPool2d(kernel_size, kernel_size)
                self._up_samples[level] = nn.Identity()
            elif self.hparams.depth_down_sample == "up":
                self._down_samples[level] = nn.Identity()
                self._up_samples[level] = nn.Upsample(scale_factor=kernel_size)
            elif self.hparams.depth_down_sample == "net":
                self._down_samples[level] = nn.Identity()
                self._up_samples[level] = nn.Identity()

    def loss(self, batch):
        images = self.get_images(batch)
        depths = []
        inner_depth_results = []
        multi_depth_results = []
        for image in images:
            depths.append(self.depth(image))
            inner_depth_results.append(self._depth_net.get_inner_result())
            if self.hparams.depth_down_sample == "net":
                multi_depth_results.append(self._depth_net.get_multi_depths())

        transformations = self.get_transformations(images, batch)

        losses = self._criterion(images, depths, transformations)
        for level, criterion in self._inner_criterions.items():
            if self.hparams.depth_down_sample == "min":
                inner_depths = [-self._down_samples[level](-x) for x in depths]
            elif self.hparams.depth_down_sample == "net":
                inner_depths = [x[level] for x in multi_depth_results]
            else:
                inner_depths = [self._down_samples[level](x) for x in depths]
            inner_images = [self._up_samples[level](x[level]) for x in inner_depth_results]
            if self.hparams.detach:
                inner_images = [x.detach() for x in inner_images]
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
        multi_depth_results = []
        for image in images:
            depths.append(self.depth(image)[0])
            inner_depth_results.append(self._depth_net.get_inner_result())
            if self.hparams.depth_down_sample == "net":
                multi_depth_results.append(self._depth_net.get_multi_depths())
        images = [image[0] for image in images]
        figure = self._result_visualizer(images, depths)
        self.logger.log_figure("depth_reconstruction", figure, self.global_step)

        for level, criterion in self._inner_criterions.items():
            if self.hparams.depth_down_sample == "min":
                inner_depths = [-self._down_samples[level](-x)[None] for x in depths]
            elif self.hparams.depth_down_sample == "net":
                inner_depths = [x[level] for x in multi_depth_results]
            else:
                inner_depths = [self._down_samples[level](x)[None] for x in depths]
            inner_images = [self._up_samples[level](x[level]) for x in inner_depth_results]
            figure = show_inner_spatial_loss(inner_images, inner_depths, criterion.get_cameras_calibration(),
                                             dpi=100)
            self.logger.log_figure(f"inner_loss_{level}", figure, self.global_step)
