import os
import sys
import unittest

import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning.utilities.parsing import AttributeDict

from deep_depth_transfer import UnsupervisedDepthModel, PoseNetResNet, DepthNetResNet, UnsupervisedCriterion
from deep_depth_transfer.data import TumValidationDataModuleFactory
from deep_depth_transfer.models import DepthEvaluationModel
from deep_depth_transfer.utils import DepthMetric
from test.data_module_mock import DataModuleMock

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class TestDepthEvaluationModel(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(os.path.dirname(current_folder), "datasets", "tum_rgbd",
                                      "rgbd_dataset_freiburg3_large_cabinet_validation")
        data_module_factory = TumValidationDataModuleFactory(dataset_folder)
        self._data_module = data_module_factory.make_data_module(
            final_image_size=(128, 384),
            batch_size=1,
            num_workers=WORKERS_COUNT,
        )
        self._data_module = DataModuleMock(self._data_module)

        depth_net = DepthNetResNet()
        self._model = DepthEvaluationModel(depth_net, DepthMetric()).cuda()

    def test_evaluation_model(self):
        tb_logger = pl.loggers.TensorBoardLogger('logs/')
        trainer = pl.Trainer(logger=tb_logger, max_epochs=1, gpus=1, progress_bar_refresh_rate=20)
        trainer.test(self._model, self._data_module.test_dataloader())
