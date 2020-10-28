from .kitti_eigen_cameras_calibration import KittiEigenCamerasCalibration
from .video_dataset_adapter import VideoDatasetAdapter
from ..data_transform_manager import DataTransformManager
from ..unsupervised_depth_data_module import UnsupervisedDepthDataModule
from ..video_dataset import VideoDataset


class KittiEigenVideoDataModuleFactory(object):
    def __init__(self, main_folder, split = "my_split"):
        self._main_folder = main_folder
        self._split = split

    def make_data_module(self, transform_manager_parameters, final_image_size, split, batch_size, num_workers,
                         device):
        video_dataset_l = VideoDatasetAdapter(self._main_folder, self._split, "l")
        video_dataset_r = VideoDatasetAdapter(self._main_folder, self._split, "r")
        original_image_size = video_dataset_l.get_image_size()
        transform_manager = DataTransformManager(
            original_image_size,
            final_image_size,
            transform_manager_parameters,
        )
        dataset = VideoDataset(
            video_dataset_l,
            video_dataset_r,
            transform = transform_manager.get_train_transform()
        )
        cameras_calibration = KittiEigenCamerasCalibration(final_image_size, original_image_size, device)
        return UnsupervisedDepthDataModule(dataset,
                                           transform_manager,
                                           cameras_calibration=cameras_calibration,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           split=split)
