import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader


# noinspection PyAbstractClass
class TumValidationDataModule(pl.LightningDataModule):
    # noinspection PyArgumentList
    def __init__(self, test_dataset, batch_size=64, num_workers=4):
        super().__init__()
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._test_dataset = test_dataset
        print(f"[Dataset] - test size = {len(self._test_dataset)}")

    def test_dataloader(self, with_normalize=False):
        return DataLoader(self._test_dataset,
                          batch_size=self._batch_size,
                          shuffle=False,
                          num_workers=self._num_workers)
