import sys
import unittest

from deep_depth_transfer.data import Downloader
from deep_depth_transfer.models import DepthNet
from deep_depth_transfer.problems import SupervisedDatasetManager, SupervisedDepthProblem
from deep_depth_transfer.utils import OptimizerManager, TrainingProcessHandler
from deep_depth_transfer.criterion import SupervisedCriterion

from deep_depth_transfer.data.supervised import GroundTruthDataset

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class DataLoaderMock(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader

    def __len__(self):
        return 1

    def __iter__(self):
        for batch in self._data_loader:
            yield batch
            break


class DatasetManagerMock(SupervisedDatasetManager):
    def get_train_batches(self, batch_size):
        return DataLoaderMock(super(DatasetManagerMock, self).get_train_batches(batch_size))

    def get_validation_batches(self, batch_size):
        return DataLoaderMock(super(DatasetManagerMock, self).get_validation_batches(batch_size))

    def get_test_batches(self, batch_size):
        return DataLoaderMock(super(DatasetManagerMock, self).get_test_batches(batch_size))


class TestSupervisedDepthProblem(unittest.TestCase):
    @unittest.skip("")
    def test_supervised_depth_problem(self):
        dataset = GroundTruthDataset(length=260)
        print (dataset._names)
        lengths = (200, 30, 30)
        dataset_manager = SupervisedDatasetManager(dataset, lengths=lengths, num_workers=WORKERS_COUNT)
        model = DepthNet(max_depth=2., min_depth=1.0).cuda()
        optimizer_manger = OptimizerManager()
        criterion = SupervisedCriterion(0.01)
        handler = TrainingProcessHandler(mlflow_tags={"name": "test"})
        problem = SupervisedDepthProblem(model, criterion, optimizer_manger, dataset_manager, handler,
                                         batch_size=1)
        problem.train(1)
