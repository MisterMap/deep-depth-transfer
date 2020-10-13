import os

from ..custom.video_dataset_adapter import VideoDatasetAdapter
from .skoltech_cameras_calibration_factory import SkoltechCamerasCalibrationFactory
from ..data_transform_manager import DataTransformManager
from ..unsupervised_depth_data_module import UnsupervisedDepthDataModule
from ..video_dataset import VideoDataset


class SkoltechDataModuleFactory():
    def __init__(self, directory="datasets"):
        self._left_directory = os.path.join(os.path.join(directory,'sequences','01','image_2'))
        self._right_directory = os.path.join(os.path.join(directory,'sequences','01','image_3'))

    def make_dataset_manager(self,
                             final_size,
                             transform_manager_parameters,
                             batch_size=64,
                             split=(80, 10, 10),
                             num_workers=4,
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
        cameras_calibration = SkoltechCamerasCalibrationFactory().make_cameras_calibration(original_image_size,
                                                                                           final_size, device)
        return UnsupervisedDepthDataModule(dataset,
                                           transform_manager,
                                           cameras_calibration,
                                           batch_size,
                                           num_workers=num_workers,
                                           split=split)
