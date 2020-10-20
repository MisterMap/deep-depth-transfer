from .tum_validation_data_module import TumValidationDataModule
from .tum_validation_dataset import TumValidationDataset
from ..concat_dataset import ConcatDataset


class TumValidationDataModuleFactory(object):
    def __init__(self, dataset_folders):
        if type(dataset_folders) is not list:
            dataset_folders = [dataset_folders]
        self._dataset_folders = dataset_folders

    def make_data_module(self, final_image_size, batch_size, num_workers):
        dataset = ConcatDataset([TumValidationDataset(x, final_image_size) for x in self._dataset_folders])
        return TumValidationDataModule(dataset, batch_size, num_workers)
