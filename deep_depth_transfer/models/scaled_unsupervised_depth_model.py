import torch
from .unsupervised_depth_model import UnsupervisedDepthModel


class ScaledUnsupervisedDepthModel(UnsupervisedDepthModel):
    def __init__(self, params, pose_net, depth_net, criterion, result_visualizer=None, *args, **kwargs):
        super().__init__(params, pose_net, depth_net, criterion, result_visualizer, *args, **kwargs)
        self._log_min_depth = torch.nn.Parameter(torch.tensor(self.hparams.initial_log_min_depth))
        self._log_scale = torch.nn.Parameter(torch.tensor(self.hparams.initial_log_scale))
        self._log_pose_scale = torch.nn.Parameter(torch.tensor(self.hparams.initial_log_pose_scale))

    def depth(self, x):
        x = (x - self._mean) / self._std
        out = self._depth_net(x, is_return_depth=False)
        out = torch.exp(self._log_min_depth) + torch.exp(self._log_scale) * torch.sigmoid(out)
        return out

    def pose(self, x, reference_frame):
        x = (x - self._mean) / self._std
        reference_frame = (reference_frame - self._mean) / self._std
        (out_rotation, out_translation) = self._pose_net(x, reference_frame)
        return torch.exp(self._log_pose_scale) * out_rotation, torch.exp(self._log_pose_scale) * out_translation

    def loss(self, batch):
        result = super().loss(batch)
        result["log_scale"] = self._log_scale
        result["log_min_depth"] = self._log_min_depth
        result["log_pose_scale"] = self._log_pose_scale
        return result

    def configure_optimizers(self):
        scale_parameters = {self._log_scale, self._log_min_depth, self._log_pose_scale}
        all_parameters = set(self.parameters())
        other_parameters = all_parameters - scale_parameters
        return torch.optim.Adam([
            {"params": list(other_parameters)},
            {"params": list(scale_parameters), "lr": self.hparams.scale_lr}],
            lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))
