import sys
import unittest

import torch
import os
from DDT.data import KittyDatasetManagerFactory

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class TestUnsupervisedDatasetManager(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(os.path.dirname(current_folder), "datasets", "kitty")
        dataset_manager_factory = KittyDatasetManagerFactory(range(0, 301, 1), directory=dataset_folder)
        self._dataset_manager = dataset_manager_factory.make_dataset_manager(
            final_size=(128, 384),
            transform_manager_parameters={"filters": True},
            num_workers=WORKERS_COUNT,
            split=(0.8, 0.1, 0.1)
        )
        self._lengths = (240, 30, 30)

    def test_dataset_manager(self):
        self.assertEqual(len(self._dataset_manager.get_train_dataset()), self._lengths[0])
        self.assertEqual(len(self._dataset_manager.get_test_dataset()), self._lengths[1])
        self.assertEqual(len(self._dataset_manager.get_validation_dataset()), self._lengths[2])
        batches = self._dataset_manager.get_train_batches(20)
        for batch in batches:
            self.assertEqual(batch["left_current_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["right_current_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["left_next_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["right_next_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["current_position"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["current_angle"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["next_position"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["next_angle"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["delta_position"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["delta_angle"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["right_next_image"].dtype, torch.float32)
            break
        batches = self._dataset_manager.get_validation_batches(20)
        for batch in batches:
            self.assertEqual(batch["left_current_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["right_current_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["left_next_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["right_next_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["current_position"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["current_angle"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["next_position"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["next_angle"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["delta_position"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["delta_angle"].shape, torch.Size([20, 3]))
            break
        batches = self._dataset_manager.get_test_batches(20)
        for batch in batches:
            self.assertEqual(batch["left_current_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["right_current_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["left_next_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["right_next_image"].shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(batch["current_position"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["current_angle"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["next_position"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["next_angle"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["delta_position"].shape, torch.Size([20, 3]))
            self.assertEqual(batch["delta_angle"].shape, torch.Size([20, 3]))
            break

    def test_get_cameras_calibration(self):
        camera_calibration = self._dataset_manager.get_cameras_calibration()
        self.assertEqual(camera_calibration.left_camera_matrix.shape, torch.Size([1, 3, 3]))
        self.assertEqual(camera_calibration.right_camera_matrix.shape, torch.Size([1, 3, 3]))
