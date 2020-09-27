from torch.utils.data import DataLoader

from DDT.utils import DatasetManager


class UnsupervisedDatasetManager(DatasetManager):
    def __init__(self, train_dataset, data_transform_manager, cameras_calibration, test_dataset=None,
                 num_workers=4, split=(80, 10, 10)):
        self._num_workers = num_workers
        super().__init__(train_dataset, test_dataset, split)
        self._data_transform_manager = data_transform_manager
        self._cameras_calibration = cameras_calibration

    def get_train_batches(self, batch_size):
        self._train_dataset.dataset.set_transform(self._data_transform_manager.get_train_transform())
        return DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=self._num_workers)

    def get_validation_batches(self, batch_size, with_normalize=False):
        self._validation_dataset.dataset.set_transform(
            self._data_transform_manager.get_validation_transform(with_normalize=with_normalize))
        return DataLoader(self._validation_dataset, batch_size=batch_size, shuffle=False, num_workers=self._num_workers)

    def get_test_batches(self, batch_size, with_normalize=False):
        self._test_dataset.dataset.set_transform(
            self._data_transform_manager.get_test_transform(with_normalize=with_normalize))
        return DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False, num_workers=self._num_workers)

    def get_validation_dataset(self, with_normalize=False):
        self._validation_dataset.dataset.set_transform(
            self._data_transform_manager.get_validation_transform(with_normalize=with_normalize))
        return self._validation_dataset

    def get_cameras_calibration(self):
        return self._cameras_calibration
