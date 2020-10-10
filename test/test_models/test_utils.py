import copy
import subprocess

import unittest

import torch
from pytorch_lightning.utilities.parsing import AttributeDict

from deep_depth_transfer import PoseNetResNet, DepthNetResNet, UnsupervisedCriterion, UnsupervisedDepthModel
from deep_depth_transfer.models import ScaledUnsupervisedDepthModel

from deep_depth_transfer.models.utils import load_undeepvo_checkpoint


class TestLoadUndeepvoCheckpointFunc(unittest.TestCase):
    def test_load_undeepvo_checkpoint(self):
        filename = 'checkpoint_undeepvo.pth'
        # subprocess.run('checkpoint_download.sh')

        params = AttributeDict(
            lr=1e-4,
            beta1=0.9,
            beta2=0.99,
            lambda_position=0.01,
            lambda_rotation=0.1,
            batch_size=8,
        )

        pose_net = PoseNetResNet()
        depth_net = DepthNetResNet()

        params.update(
            scale_lr=5e-1,
            initial_log_scale=4.59,
            initial_log_min_depth=0.
        )

        model = ScaledUnsupervisedDepthModel(
            params,
            pose_net,
            depth_net,
            criterion=None
        )

        model_before = copy.deepcopy(model)
        load_undeepvo_checkpoint(model, filename)
        self.assertTrue(torch.any(model_before._pose_net._first_layer.weight != model._pose_net._first_layer.weight))
        self.assertTrue(torch.any(model_before._depth_net.skip_zero.weight != model._depth_net.skip_zero.weight))
