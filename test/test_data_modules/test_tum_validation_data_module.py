import sys
import unittest

import torch
import os
from deep_depth_transfer.data import TumValidationDataModuleFactory

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class TestTumValidationDataModule(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(os.path.dirname(current_folder), "datasets", "tum_rgbd",
                                      "rgbd_dataset_freiburg3_large_cabinet_validation")
        data_module_factory = TumValidationDataModuleFactory(dataset_folder)
        self._data_module = data_module_factory.make_data_module(
            final_image_size=(128, 384),
            batch_size=20,
            num_workers=WORKERS_COUNT,
        )
        self._test_dataset_length = 1012

    def test_data_module(self):
        self.assertEqual(len(self._data_module._test_dataset), self._test_dataset_length)
        batches = self._data_module.test_dataloader()
        for batch in batches:
            self.assertEqual(batch["image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["ground_truth_depth"].shape, torch.Size([20, 1, 213, 640]))
            break

