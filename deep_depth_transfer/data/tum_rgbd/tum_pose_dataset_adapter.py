import os.path
import scipy.spatial.transform
import scipy.interpolate
from ..pose_data_point import PoseDataPoint
import numpy as np


class TumPoseDatasetAdapter(object):
    def __init__(self, main_folder):
        with open(os.path.join(main_folder, "rgb.txt")) as fd:
            self._rgb_images = fd.read().splitlines()
            self._rgb_timestamps = [float(element.split(" ")[0]) for element in self._rgb_images[3:]]
        self._main_folder = main_folder
        with open(os.path.join(main_folder, "groundtruth.txt")) as fd:
            self._ground_truth = fd.read().splitlines()
            self._ground_truth = [element.split(" ") for element in self._ground_truth[3:]]
        times = np.array([float(x[0]) for x in self._ground_truth])
        positions = np.array([[float(x) for x in y[1:]] for y in self._ground_truth])
        self._position_interpolation = scipy.interpolate.interp1d(times, positions, kind="linear",
                                                                  fill_value="extrapolate", axis=0)
        self._length = len(self._rgb_images)

    def __getitem__(self, index):
        return PoseDataPoint(self.transformation_matrix(index), self.transformation_matrix(index + 1)).get_data()

    def transformation_matrix(self, index):
        timestamp = self._rgb_timestamps[index]
        position = self._position_interpolation(timestamp)
        transformation = np.eye(4, dtype=np.float32)
        rotation_matrix = scipy.spatial.transform.Rotation.from_quat(position[3:]).as_matrix()
        transformation[:3, :3] = rotation_matrix
        transformation[:3, 3] = position[:3]
        return transformation

    def __len__(self):
        return self._length
