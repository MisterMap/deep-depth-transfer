import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader


# noinspection PyAbstractClass
class UnsupervisedDepthDataModule(pl.LightningDataModule):
    # noinspection PyArgumentList
    def __init__(self, train_dataset, data_transform_manager, cameras_calibration, batch_size=64,
                 test_dataset=None,
                 num_workers=4, split=(80, 10, 10)):
        super().__init__()
        self._num_workers = num_workers
        self._data_transform_manager = data_transform_manager
        self._cameras_calibration = cameras_calibration
        self._batch_size = batch_size

        train_length = len(train_dataset)
        if test_dataset is None:
            lengths = int(train_length * split[0]), int(train_length * split[1]), \
                train_length - int(train_length * split[0]) - int(train_length * split[1])
            self._train_dataset, self._validation_dataset, self._test_dataset = \
                torch.utils.data.random_split(train_dataset, lengths)
        else:
            lengths = int(train_length * split[0]), train_length - int(train_length * split[0])
            self._train_dataset, self._validation_dataset = \
                torch.utils.data.random_split(train_dataset, lengths)
            self._test_dataset = test_dataset
        print(f"[Dataset] - train size = {len(self._train_dataset)}")
        print(f"[Dataset] - validation size = {len(self._validation_dataset)}")
        print(f"[Dataset] - test size = {len(self._test_dataset)}")

    def train_dataloader(self):
        self._train_dataset.dataset.set_transform(self._data_transform_manager.get_train_transform())
        return DataLoader(self._train_dataset,
                          batch_size=self._batch_size,
                          shuffle=True,
                          num_workers=self._num_workers)

    def val_dataloader(self, normalize=False):
        self._validation_dataset.dataset.set_transform(
            self._data_transform_manager.get_validation_transform(with_normalize=normalize))
        return DataLoader(self._validation_dataset,
                          batch_size=self._batch_size,
                          shuffle=False,
                          num_workers=self._num_workers)

    def test_dataloader(self, normalize=False):
        self._test_dataset.dataset.set_transform(
            self._data_transform_manager.get_test_transform(with_normalize=normalize))
        return DataLoader(self._test_dataset,
                          batch_size=self._batch_size,
                          shuffle=False,
                          num_workers=self._num_workers)

    def get_cameras_calibration(self):
        return self._cameras_calibration

    def test_dataset(self, parameters=False):
        transform = self._data_transform_manager.get_test_transform(parameters)
        self._test_dataset.dataset.set_transform(transform)
        return self._test_dataset

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
