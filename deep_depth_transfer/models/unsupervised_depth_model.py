import pytorch_lightning as pl
import torch
import torch.nn as nn


class UnsupervisedDepthModel(pl.LightningModule):
    def __init__(self, params, pose_net, depth_net, criterion, result_visualizer=None, stereo=True, mono=True,
                 use_ground_truth_poses=False):
        super().__init__()
        self.save_hyperparameters(params)
        self.hparams.model = str(type(self))
        assert stereo or mono
        self._stereo = stereo
        self._mono = mono
        self._use_groun_truth_poses = use_ground_truth_poses
        self._pose_net = pose_net
        self._depth_net = depth_net
        self._criterion = criterion
        self._result_visualizer = result_visualizer
        self.example_input_array = (
            torch.zeros((1, 3, 128, 384), dtype=torch.float),
            torch.zeros((1, 3, 128, 384), dtype=torch.float))
        self._mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self._std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def cuda(self, *args, **kwargs):
        self._mean = self._mean.cuda(*args, **kwargs)
        self._std = self._std.cuda(*args, **kwargs)
        self._depth_net = self._depth_net.cuda(*args, **kwargs)
        self._pose_net = self._pose_net.cuda(*args, **kwargs)
        return super().to(*args, **kwargs)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def depth(self, x):
        x = (x - self._mean) / self._std
        out = self._depth_net(x)
        return out

    def pose(self, x, reference_frame):
        x = (x - self._mean) / self._std
        reference_frame = (reference_frame - self._mean) / self._std
        (out_rotation, out_translation) = self._pose_net(x, reference_frame)
        return out_rotation, out_translation

    def forward(self, x, reference_frame):
        return self.depth(x), self.pose(x, reference_frame)

    def get_images(self, batch):
        # current_left, next_left, current_right, next_right
        if self._mono and self._stereo:
            images = [
                batch["left_current_image"],
                batch["left_next_image"],
                batch["right_current_image"],
                batch["right_next_image"]
            ]
        elif self._mono:
            images = [
                batch["current_image"],
                batch["next_image"]
            ]
        else:
            images = [
                batch["left_image"],
                batch["right_image"]
            ]
        return images

    def get_transformations(self, images, batch):
        if self._mono and self._stereo and not self._use_groun_truth_poses:
            transformations = [
                self.pose(images[0], images[1]),
                self.pose(images[1], images[0]),
                self.pose(images[2], images[3]),
                self.pose(images[3], images[2])
            ]
        elif self._mono and not self._use_groun_truth_poses:
            transformations = [
                self.pose(images[0], images[1]),
                self.pose(images[1], images[0])
            ]
        elif self._mono and not self._stereo:
            transformations = [
                (batch["current_angle"], batch["current_position"]),
                (batch["next_angle"], batch["next_position"]),
            ]
        else:
            transformations = None
        return transformations

    def loss(self, batch):
        images = self.get_images(batch)
        depths = [self.depth(image) for image in images]
        transformations = self.get_transformations(images, batch)
        return self._criterion(images, depths, transformations)

    def training_step(self, batch, *args):
        losses = self.loss(batch)
        train_losses = {}
        for key, value in losses.items():
            train_losses[f"train_{key}"] = value
        self.log_dict(train_losses)
        return losses["loss"]

    def show_figures(self, batch, batch_index):
        if self._result_visualizer is None:
            return
        if not self._result_visualizer.batch_index == batch_index:
            return
        images = self.get_images(batch)
        depths = [self.depth(image[:1])[0] for image in images]
        images = [image[0] for image in images]
        figure = self._result_visualizer(images, depths)
        self.logger.log_figure("depth_reconstruction", figure, self.global_step)

    def validation_step(self, batch, batch_index: int):
        losses = self.loss(batch)

        self.show_figures(batch, batch_index)

        val_losses = {}
        for key, value in losses.items():
            val_losses[f"val_{key}"] = value
        self.log_dict(val_losses)
        return losses["loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.lr,
                                betas=(self.hparams.beta1, self.hparams.beta2))
