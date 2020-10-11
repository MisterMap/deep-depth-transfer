import os
import sys
import unittest

import pytorch_lightning as pl
import pytorch_lightning.loggers

from deep_depth_transfer import ResultVisualizer
from deep_depth_transfer import UnsupervisedDepthModel, PoseNetResNet, DepthNetResNet, UnsupervisedCriterion
from deep_depth_transfer.data import KittiDataModuleFactory
from test.data_module_mock import DataModuleMock
from deep_depth_transfer.utils import LoggerCollection, TensorBoardLogger, MLFlowLogger
from pytorch_lightning.utilities import AttributeDict

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class TestResultVisualizer(unittest.TestCase):
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
        params = AttributeDict(lr=1e-3, beta1=0.99, beta2=0.9)
        self._model = UnsupervisedDepthModel(params, pose_net, depth_net, criterion,
                                             result_visualizer=result_visualizer).cuda()
        self._tb_logger = TensorBoardLogger('logs/')
        self._second_tb_logger = TensorBoardLogger('logs1/')
        self._double_tb_logger = LoggerCollection([self._tb_logger, self._second_tb_logger])

        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://ec2-3-134-104-174.us-east-2.compute.amazonaws.com:9000"
        os.environ["AWS_ACCESS_KEY_ID"] = "depth"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "depth123"
        self._mlflow_logger = MLFlowLogger(experiment_name="test",
            tracking_uri="http://ec2-3-134-104-174.us-east-2.compute.amazonaws.com:5001")

    def test_tb_logger(self):
        trainer = pl.Trainer(logger=self._tb_logger, max_epochs=1, gpus=1, progress_bar_refresh_rate=20)
        trainer.fit(self._model, self._data_module)

    def test_double_tb_logger(self):
        trainer = pl.Trainer(logger=self._double_tb_logger,
                             max_epochs=1, gpus=1, progress_bar_refresh_rate=20)
        trainer.fit(self._model, self._data_module)

    def test_mlflow_logger(self):
        trainer = pl.Trainer(logger=self._mlflow_logger,
                             max_epochs=1, gpus=1, progress_bar_refresh_rate=20)
        trainer.fit(self._model, self._data_module)