import os
import sys
import unittest

import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict

from deep_depth_transfer.utils import TensorBoardLogger
from deep_depth_transfer import PoseNetResNet, DepthNetResNet, UnsupervisedCriterion
from deep_depth_transfer.models import MultiUnsupervisedDepthModel
from deep_depth_transfer.data import KittiDataModuleFactory
from test.data_module_mock import DataModuleMock
from deep_depth_transfer import ResultVisualizer

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class TestUnsupervisedDepthModel(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(os.path.dirname(current_folder), "datasets", "kitti")
        data_module_factory = KittiDataModuleFactory(range(0, 301, 1), directory=dataset_folder)
        self._data_module = data_module_factory.make_dataset_manager(
            final_image_size=(128, 384),
            transform_manager_parameters={"filters": True},
            batch_size=1,
            num_workers=WORKERS_COUNT,
            split=(0.8, 0.1, 0.1)
        )
        self._data_module = DataModuleMock(self._data_module)

        pose_net = PoseNetResNet()
        depth_net = DepthNetResNet()
        criterion = UnsupervisedCriterion(self._data_module.get_cameras_calibration(), 1, 1)

        result_visualizer = ResultVisualizer(cameras_calibration=self._data_module.get_cameras_calibration())

        inner_criterions = {}
        levels = [2, 3]
        for level in levels:
            cameras_calibration = self._data_module.get_cameras_calibration().calculate_scaled_cameras_calibration(
                scale=level ** 2,
                image_size=(128, 384)
            )
            inner_criterions[level] = UnsupervisedCriterion(cameras_calibration, lambda_s=0.15, smooth_loss=False,
                                                            pose_loss=False)
        params = AttributeDict(lr=1e-3, beta1=0.99, beta2=0.9)
        self._model = MultiUnsupervisedDepthModel(params, pose_net, depth_net, criterion,
                                                  inner_criterions=inner_criterions,
                                                  result_visualizer=result_visualizer)

    def test_unsupervised_depth_model(self):
        logger = TensorBoardLogger("lightning_logs")
        trainer = pl.Trainer(logger=logger, max_epochs=1, gpus=1, progress_bar_refresh_rate=20)
        trainer.fit(self._model, self._data_module)
