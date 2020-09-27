from .data_loader_mock import DataLoaderMock


class DatasetManagerMock(object):
    def __init__(self, dataset_manager):
        self._dataset_manager = dataset_manager

    def get_train_batches(self, batch_size):
        return DataLoaderMock(self._dataset_manager.get_train_batches(batch_size))

    def get_validation_batches(self, batch_size, **_):
        return DataLoaderMock(self._dataset_manager.get_validation_batches(batch_size))

    def get_test_batches(self, batch_size, **_):
        return DataLoaderMock(self._dataset_manager.get_test_batches(batch_size))

    def get_validation_dataset(self, *args, **kwargs):
        return self._dataset_manager.get_validation_dataset(*args, **kwargs)

    def get_cameras_calibration(self, *args, **kwargs):
        return self._dataset_manager.get_cameras_calibration(*args, **kwargs)
