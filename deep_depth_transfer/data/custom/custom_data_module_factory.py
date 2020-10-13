import os

from .custom_cameras_calibration_factory import CustomCamerasCalibrationFactory
from .video_dataset_adapter import VideoDatasetAdapter
from ..data_transform_manager import DataTransformManager
from ..unsupervised_depth_data_module import UnsupervisedDepthDataModule
from ..video_dataset import VideoDataset


class CustomDataModuleFactory():
    def __init__(self, directory="datasets"):
        self._left_directory = os.path.join(directory, "left")
        self._right_directory = os.path.join(directory, "right")

    def make_dataset_manager(self, final_size, transform_manager_parameters, split=(80, 10, 10), num_workers=4,
                             device="cpu"):
        left_dataset = VideoDatasetAdapter(self._left_directory)
        right_dataset = VideoDatasetAdapter(self._right_directory)
        original_image_size = left_dataset.get_image_size()
        transform_manager = DataTransformManager(
            original_image_size,
            final_size,
            transform_manager_parameters
        )
        dataset = VideoDataset(
            left_dataset,
            right_dataset
        )
        cameras_calibration = CustomCamerasCalibrationFactory().make_cameras_calibration(original_image_size,
                                                                                         final_size, device)
        return UnsupervisedDepthDataModule(dataset, transform_manager, cameras_calibration,
                                          num_workers=num_workers, split=split)
