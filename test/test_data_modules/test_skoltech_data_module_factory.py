import sys
import unittest

import torch
import torchvision
import os
from deep_depth_transfer.data import SkoltechDataModuleFactory

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


# Need Skoltech dataset
class TestSkoltechDataModule(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(os.path.dirname(current_folder), "datasets", "skoltech")
        data_module_factory = SkoltechDataModuleFactory(directory=dataset_folder)
        self._batch_size = 10
        self._dataset_size = 19
        self._split = (0.8, 0.2, 0.0)
        self._data_module = data_module_factory.make_dataset_manager(
            final_size=(128, 384),
            transform_manager_parameters={"filters": True},
            batch_size=self._batch_size,
            num_workers=WORKERS_COUNT,
            split=self._split
        )
        self._lengths = [int(x * self._dataset_size) for x in self._split]

    def test_data_module(self):
        self.assertEqual(len(self._data_module._train_dataset), self._lengths[0])
        self.assertEqual(len(self._data_module._validation_dataset), self._lengths[1])
        self.assertEqual(len(self._data_module._test_dataset), self._lengths[2] + 1)
        batches = self._data_module.train_dataloader()

        for i, batch in enumerate(batches):
            self.assertEqual(batch["left_current_image"].shape, torch.Size([self._batch_size, 3, 128, 384]))
            self.assertEqual(batch["right_current_image"].shape, torch.Size([self._batch_size, 3, 128, 384]))
            self.assertEqual(batch["left_next_image"].shape, torch.Size([self._batch_size, 3, 128, 384]))
            self.assertEqual(batch["right_next_image"].shape, torch.Size([self._batch_size, 3, 128, 384]))
            self.assertEqual(batch["right_next_image"].dtype, torch.float32)
            if i == 0:
                img = batch["left_current_image"][0].cpu()
                img = torchvision.transforms.ToPILImage()(img)
                img.save('skoltech_dataset_img.png')
            break

        batches = self._data_module.val_dataloader()
        for batch in batches:
            self.assertEqual(batch["left_current_image"].shape, torch.Size([self._lengths[1], 3, 128, 384]))
            self.assertEqual(batch["right_current_image"].shape, torch.Size([self._lengths[1], 3, 128, 384]))
            self.assertEqual(batch["left_next_image"].shape, torch.Size([self._lengths[1], 3, 128, 384]))
            self.assertEqual(batch["right_next_image"].shape, torch.Size([self._lengths[1], 3, 128, 384]))
            break
        batches = self._data_module.test_dataloader()
        for batch in batches:
            self.assertEqual(batch["left_current_image"].shape, torch.Size([self._lengths[2] + 1, 3, 128, 384]))
            self.assertEqual(batch["right_current_image"].shape, torch.Size([self._lengths[2] + 1, 3, 128, 384]))
            self.assertEqual(batch["left_next_image"].shape, torch.Size([self._lengths[2] + 1, 3, 128, 384]))
            self.assertEqual(batch["right_next_image"].shape, torch.Size([self._lengths[2] + 1, 3, 128, 384]))
            break

    def test_get_cameras_calibration(self):
        camera_calibration = self._data_module.get_cameras_calibration()
        self.assertEqual(camera_calibration.left_camera_matrix.shape, torch.Size([1, 3, 3]))
        self.assertEqual(camera_calibration.right_camera_matrix.shape, torch.Size([1, 3, 3]))
