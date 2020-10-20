from .tum_cameras_calibration import TumCamerasCalibration
from .video_dataset_adapter import VideoDatasetAdapter
from ..data_transform_manager import DataTransformManager
from ..unsupervised_depth_data_module import UnsupervisedDepthDataModule
from ..video_dataset import VideoDataset
from ..concat_dataset import ConcatDataset


class TumVideoDataModuleFactory(object):
    def __init__(self, main_folders):
        if type(main_folders) is not list:
            main_folders = [main_folders]
        self._main_folders = main_folders

    def make_data_module(self, transform_manager_parameters, final_image_size, split, batch_size, num_workers,
                         device):
        datasets = [VideoDataset(VideoDatasetAdapter(x)) for x in self._main_folders]
        original_image_size = VideoDatasetAdapter(self._main_folders[0]).get_image_size()
        transform_manager = DataTransformManager(
            original_image_size,
            final_image_size,
            transform_manager_parameters,
            custom_additional_targets={"image2": "image"}
        )
        dataset = ConcatDataset(datasets)
        cameras_calibration = TumCamerasCalibration(final_image_size, original_image_size, device)
        return UnsupervisedDepthDataModule(dataset,
                                           transform_manager,
                                           cameras_calibration=cameras_calibration,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           split=split)
