import os
import sys
import unittest

import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.parsing import Namespace
import pytorch_lightning.core.saving

from deep_depth_transfer.data import KittiDataModuleFactory
from deep_depth_transfer.models.factory import MultiUnsupervisedDepthModelFactory
from deep_depth_transfer.utils import TensorBoardLogger
from test.data_module_mock import DataModuleMock

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class TestUnsupervisedDepthModel(unittest.TestCase):
    def setUp(self) -> None:
        params = AttributeDict(
            image_size=(128, 384),
            batch_size=1,
            transform_filters=True,
            split=(0.8, 0.1, 0.1),
            num_workers=WORKERS_COUNT,
            detach=True,
            levels=(1,),
            inner_lambda_s=0.15,
            lr=1e-3,
            beta1=0.99,
            beta2=0.9
        )
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(os.path.dirname(current_folder), "datasets", "kitti")
        data_module_factory = KittiDataModuleFactory(range(0, 301, 1), directory=dataset_folder)
        self._data_module = data_module_factory.make_data_module_from_params(params)
        self._data_module = DataModuleMock(self._data_module)

        self._model = MultiUnsupervisedDepthModelFactory().make_model(
            params,
            self._data_module.get_cameras_calibration()
        )

    # noinspection PyTypeChecker
    def test_unsupervised_depth_model(self):
        logger = TensorBoardLogger("lightning_logs")
        trainer = pl.Trainer(logger=logger, max_epochs=1, gpus=1, progress_bar_refresh_rate=20)
        trainer.fit(self._model, self._data_module)
