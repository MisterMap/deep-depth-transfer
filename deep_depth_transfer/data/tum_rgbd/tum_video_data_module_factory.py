from .tum_cameras_calibration import TumCamerasCalibration
from .video_dataset_adapter import VideoDatasetAdapter
from ..data_transform_manager import DataTransformManager
from ..unsupervised_depth_data_module import UnsupervisedDepthDataModule
from ..video_dataset import VideoDataset


class TumVideoDataModuleFactory(object):
    def __init__(self, main_folder):
        self._main_folder = main_folder

    def make_data_module(self, transform_manager_parameters, final_image_size, split, batch_size, num_workers,
                         device):
        video_dataset = VideoDatasetAdapter(self._main_folder)
        original_image_size = video_dataset.get_image_size()
        transform_manager = DataTransformManager(
            original_image_size,
            final_image_size,
            transform_manager_parameters,
            custom_additional_targets={"image2": "image"}
        )
        dataset = VideoDataset(
            video_dataset
        )
        cameras_calibration = TumCamerasCalibration(final_image_size, original_image_size, device)
        return UnsupervisedDepthDataModule(dataset,
                                           transform_manager,
                                           cameras_calibration=cameras_calibration,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           split=split)
