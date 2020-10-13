from .tum_validation_data_module import TumValidationDataModule
from .tum_validation_dataset import TumValidationDataset


class TumValidationDataModuleFactory(object):
    def __init__(self, dataset_folder):
        self._dataset_folder = dataset_folder

    def make_data_module(self, final_image_size, batch_size, num_workers):
        dataset = TumValidationDataset(self._dataset_folder, final_image_size)
        return TumValidationDataModule(dataset, batch_size, num_workers)
