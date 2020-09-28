import os
import os.path

import pykitti

from .data import Downloader
from .kitty_cameras_calibration_factory import KittyCamerasCalibrationFactory
from .poses_dataset_adapter import PosesDatasetAdapter
from .video_dataset_adapter import VideoDatasetAdapter
from ..data_transform_manager import DataTransformManager
from ..video_dataset import VideoDataset
from ..unsupervised_dataset_manager import UnsupervisedDatasetManager


class KittyDatasetManagerFactory():
    def __init__(self, frames, sequence="08", directory="datasets", download=False):
        sequence = Downloader(sequence, directory)
        if download and not os.path.exists(os.path.join(directory, "poses")):
            print("Download datasets")
            sequence.download_sequence()
        self._kitty_dataset = pykitti.odometry(sequence.main_dir, sequence.sequence_id, frames=frames)

    def make_dataset_manager(self, final_size, transform_manager_parameters, split=(80, 10, 10), num_workers=4,
                             device="cpu"):
        original_image_size = self._kitty_dataset.get_rgb(0)[0].size[::-1]
        transform_manager = DataTransformManager(
            original_image_size,
            final_size,
            transform_manager_parameters
        )
        dataset = VideoDataset(
            VideoDatasetAdapter(self._kitty_dataset, 0),
            VideoDatasetAdapter(self._kitty_dataset, 1),
            PosesDatasetAdapter(self._kitty_dataset)
        )
        cameras_calibration = KittyCamerasCalibrationFactory().make_cameras_calibration(original_image_size,
                                                                                        final_size, device)
        return UnsupervisedDatasetManager(dataset, transform_manager, cameras_calibration,
                                          num_workers=num_workers, split=split)
