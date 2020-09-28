import os
import sys
import unittest

import pykitti.odometry

from DDT.criterion import UnsupervisedCriterion
from DDT.data import KittyDatasetManagerFactory
from DDT.models import UnDeepVO
from DDT.problems import UnsupervisedDepthProblem
from DDT.utils import OptimizerManager, TrainingProcessHandler
from test.dataset_manager_mock import DatasetManagerMock

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class TestUnsupervisedDepthProblem(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(current_folder, "datasets", "kitty")
        dataset_manager_factory = KittyDatasetManagerFactory(range(0, 301, 1), directory=dataset_folder)
        dataset_manager = dataset_manager_factory.make_dataset_manager(
            final_size=(128, 384),
            transform_manager_parameters={"filters": True},
            num_workers=WORKERS_COUNT,
            split=(0.8, 0.1, 0.1),
            device="cuda:0"
        )
        dataset_manager = DatasetManagerMock(dataset_manager)
        model = UnDeepVO(max_depth=2., min_depth=1.0).cuda()
        optimizer_manger = OptimizerManager()
        criterion = UnsupervisedCriterion(dataset_manager.get_cameras_calibration(),
                                          0.1, 1, 0.85)
        handler = TrainingProcessHandler(mlflow_tags={"name": "test"})
        self._problem = UnsupervisedDepthProblem(model, criterion, optimizer_manger, dataset_manager, handler,
                                                 batch_size=1)

    def test_unsupervised_depth_problem(self):
        self._problem.train(1)

    def test_unsupervised_depth_problem_truth_positions(self):
        sequence_8 = Downloader('08')
        if not os.path.exists("datasets/poses"):
            print("Download datasets")
            sequence_8.download_sequence()
        lengths = (200, 30, 30)
        dataset = pykitti.odometry(sequence_8.main_dir, sequence_8.sequence_id, frames=range(0, 260, 1))
        dataset_manager = DatasetManagerMock(dataset, lenghts=lengths, num_workers=WORKERS_COUNT)
        model = UnDeepVO(max_depth=2., min_depth=1.0).cuda()
        optimizer_manger = OptimizerManager()
        criterion = UnsupervisedCriterion(dataset_manager.get_cameras_calibration("cuda:0"),
                                          0.1, 1, 0.85)
        handler = TrainingProcessHandler(mlflow_tags={"name": "test"})
        problem = UnsupervisedDepthProblem(model, criterion, optimizer_manger, dataset_manager, handler,
                                           batch_size=1, use_truth_poses=True)
        problem.train(1)

    def test_unsupervised_depth_problem_cpu(self):
        device = "cpu"
        sequence_8 = Downloader('08')
        if not os.path.exists("datasets/poses"):
            print("Download datasets")
            sequence_8.download_sequence()
        lengths = (200, 30, 30)
        dataset = pykitti.odometry(sequence_8.main_dir, sequence_8.sequence_id, frames=range(0, 260, 1))
        dataset_manager = DatasetManagerMock(dataset, lenghts=lengths, num_workers=WORKERS_COUNT)
        model = UnDeepVO(max_depth=2., min_depth=1.0).to(device)
        optimizer_manger = OptimizerManager()
        criterion = UnsupervisedCriterion(dataset_manager.get_cameras_calibration(device),
                                          0.1, 1, 0.85)
        handler = TrainingProcessHandler(mlflow_tags={"name": "test"})
        problem = UnsupervisedDepthProblem(model, criterion, optimizer_manger, dataset_manager, handler,
                                           batch_size=1, device=device)
        problem.train(1)
