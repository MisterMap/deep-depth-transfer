import pytorch_lightning as pl


class UnsupervisedDepthModel(pl.LightningModule):
    def __init__(self, pose_net, depth_net, criterion, depth_visualizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pose_net = pose_net
        self._depth_net = depth_net
        self._criterion = criterion
        self._depth_visualizer = depth_visualizer

    def depth(self, x):
        out = self.depth_net(x)
        return out

    def pose(self, x, reference_frame):
        (out_rotation, out_translation) = self.pose_net(x, reference_frame)
        return out_rotation, out_translation

    def forward(self, x, reference_frame):
        return self.depth(x), self.pose(x, reference_frame)

    def loss(self, batch):
        # current_left, next_left, current_right, next_right
        images = batch[:4]
        depths = [self.depth(image) for image in images]
        transformations = [
            self.pose(images[0], images[1]),
            self.pose(images[1], images[0]),
            self.pose(images[2], images[3]),
            self.pose(images[3], images[2])
        ]
        return self._criterion(images, depths, transformations)

    def training_step(self, batch):
        losses = self.loss(batch)
        result = pl.TrainResult(losses["loss"])
        for key, value in losses.items():
            result.log(key, value, on_step=True)

        return result

    def validation_step(self, batch, batch_index: int):
        losses = self.loss(batch)

        if batch_index == 0 and self._depth_visualizer is not None:
            self.logger.experiment.add_image(self._depth_visualizer(batch[:1]))

        result = pl.EvalResult(checkpoint_on=losses["loss"])
        for key, value in losses.items():
            result.log(key, value)
        return result
