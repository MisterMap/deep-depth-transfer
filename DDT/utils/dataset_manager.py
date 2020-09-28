import abc
from torch.utils.data import random_split


# noinspection PyUnusedLocal
class DatasetManager(abc.ABC):
    def __init__(self, train_dataset, test_dataset=None, split=(80, 10, 10)):
        length = len(train_dataset)
        if test_dataset is None:
            lengths = int(length * split[0]), int(length * split[1]), \
                      length - int(length * split[0]) - int(length * split[1])
            self._train_dataset, self._validation_dataset, self._test_dataset = random_split(train_dataset, lengths)
        else:
            lengths = int(length * split[0]), length - int(length * split[0])
            self._train_dataset, self._validation_dataset = random_split(train_dataset, split)
            self._test_dataset = test_dataset
        print(f"[Dataset] - train size = {len(self._train_dataset)}")
        print(f"[Dataset] - validation size = {len(self._validation_dataset)}")
        print(f"[Dataset] - test size = {len(self._test_dataset)}")

    @abc.abstractmethod
    def get_train_batches(self, batch_size):
        return []

    @abc.abstractmethod
    def get_validation_batches(self, batch_size):
        return []

    @abc.abstractmethod
    def get_test_batches(self, batch_size):
        return []

    def get_test_dataset(self):
        return self._test_dataset

    def get_train_dataset(self):
        return self._train_dataset

    def get_validation_dataset(self):
        return self._validation_dataset
