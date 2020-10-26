import pykitti

from .data import Downloader
from .kitti_cameras_calibration_factory import KittyCamerasCalibrationFactory
from .poses_dataset_adapter import PosesDatasetAdapter
from .video_dataset_adapter import VideoDatasetAdapter
from ..concat_dataset import ConcatDataset
from ..data_transform_manager import DataTransformManager
from ..unsupervised_depth_data_module import UnsupervisedDepthDataModule
from ..video_dataset import VideoDataset


class KittiDataModuleFactory():
    def __init__(self, frames, sequences="08", directory="datasets"):
        if type(sequences) != list:
            sequences = [sequences]
        self._kitti_datasets = [self.make_kitti_dataset(directory, x, frames) for x in sequences]

    @staticmethod
    def make_kitti_dataset(directory, sequence, frames):
        sequence = Downloader(sequence, directory)
        return pykitti.odometry(sequence.main_dir, sequence.sequence_id, frames=frames)

    @staticmethod
    def make_video_dataset(kitti_dataset):
        return VideoDataset(
            VideoDatasetAdapter(kitti_dataset, 0),
            VideoDatasetAdapter(kitti_dataset, 1),
            PosesDatasetAdapter(kitti_dataset)
        )

    def make_dataset_manager(self,
                             final_image_size,
                             transform_manager_parameters,
                             batch_size=64,
                             split=(80, 10, 10),
                             num_workers=4,
                             device="cpu"):
        original_image_size = self._kitti_datasets[0].get_rgb(0)[0].size[::-1]
        transform_manager = DataTransformManager(
            original_image_size,
            final_image_size,
            transform_manager_parameters
        )
        dataset = ConcatDataset([self.make_video_dataset(x) for x in self._kitti_datasets])
        cameras_calibration = KittyCamerasCalibrationFactory().make_cameras_calibration(
            original_image_size,
            final_image_size,
            device
        )
        return UnsupervisedDepthDataModule(dataset,
                                           transform_manager,
                                           cameras_calibration,
                                           batch_size,
                                           num_workers=num_workers,
                                           split=split)

    def make_data_module_from_params(self, params):
        transform_manager_parameters = {
            "filters": params.transform_filters
        }
        return self.make_dataset_manager(params.image_size, transform_manager_parameters,
                                         params.batch_size, params.split, params.num_workers)
