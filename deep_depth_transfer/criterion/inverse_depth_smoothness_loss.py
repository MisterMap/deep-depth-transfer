import torch
import kornia


class InverseDepthSmoothnessLoss(torch.nn.Module):
    def __init__(self, lambda_depth=1.0):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.inverse_depth_smoothness_loss = kornia.losses.InverseDepthSmoothnessLoss()

    def forward(self, depth, image):
        loss = self.inverse_depth_smoothness_loss(1.0 / depth, image)

        return self.lambda_depth * loss
