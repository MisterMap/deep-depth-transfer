from ..pose_data_point import PoseDataPoint


class PosesDatasetAdapter(object):
    def __init__(self, dataset):
        self._length = len(dataset.poses) - 1
        self._dataset = dataset

    def __getitem__(self, index):
        return PoseDataPoint(self._dataset.poses[index], self._dataset.poses[index + 1]).get_data()

    def __len__(self):
        return self._length
