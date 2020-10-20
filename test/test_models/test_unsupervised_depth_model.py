import os
import sys
import unittest

import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning.utilities.parsing import AttributeDict

from deep_depth_transfer import UnsupervisedDepthModel, PoseNetResNet, DepthNetResNet, UnsupervisedCriterion
from deep_depth_transfer.data import KittiDataModuleFactory
from test.data_module_mock import DataModuleMock

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

        params = AttributeDict(lr=1e-3, beta1=0.99, beta2=0.9)
        self._model = UnsupervisedDepthModel(params, pose_net, depth_net, criterion).cuda()

    def test_unsupervised_depth_model(self):
        tb_logger = pl.loggers.TensorBoardLogger('logs/')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath="checkpoints")
        trainer = pl.Trainer(logger=tb_logger, max_epochs=1, gpus=1, progress_bar_refresh_rate=20,
                             checkpoint_callback=checkpoint_callback)
        trainer.fit(self._model, self._data_module)
