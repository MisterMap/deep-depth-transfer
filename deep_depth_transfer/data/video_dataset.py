import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self,
                 left_video_dataset,
                 right_video_dataset=None,
                 pose_dataset=None,
                 transform=None,
                 mono_video=True):
        self._left_video_dataset = left_video_dataset
        self._right_video_dataset = right_video_dataset
        self._pose_dataset = pose_dataset
        self._transform = transform
        self._mono_video = mono_video

    def set_transform(self, transform):
        self._transform = transform

    def get_image_size(self):
        return self._left_video_dataset.get_image_size()

    def stereo_video_item(self, index):
        image_data_point = {
            "image": np.array(self._left_video_dataset[index]),
            "image2": np.array(self._right_video_dataset[index]),
            "image3": np.array(self._left_video_dataset.get_next_image(index)),
            "image4": np.array(self._right_video_dataset.get_next_image(index))
        }
        image_data_point = self._transform(**image_data_point)
        image_data_point = {
            "left_current_image": torch.from_numpy(image_data_point["image"]).permute(2, 0, 1),
            "right_current_image": torch.from_numpy(image_data_point["image2"]).permute(2, 0, 1),
            "left_next_image": torch.from_numpy(image_data_point["image3"]).permute(2, 0, 1),
            "right_next_image": torch.from_numpy(image_data_point["image4"]).permute(2, 0, 1),
        }
        return image_data_point

    def mono_video_item(self, index):
        image_data_point = {
            "image": np.array(self._left_video_dataset[index]),
            "image2": np.array(self._left_video_dataset.get_next_image(index)),
        }
        image_data_point = self._transform(**image_data_point)
        image_data_point = {
            "current_image": torch.from_numpy(image_data_point["image"]).permute(2, 0, 1),
            "next_image": torch.from_numpy(image_data_point["image2"]).permute(2, 0, 1),
        }
        return image_data_point

    def stereo_item(self, index):
        image_data_point = {
            "image": np.array(self._left_video_dataset[index]),
            "image2": np.array(self._right_video_dataset[index]),
        }
        image_data_point = self._transform(**image_data_point)
        image_data_point = {
            "left_image": torch.from_numpy(image_data_point["image"]).permute(2, 0, 1),
            "right_image": torch.from_numpy(image_data_point["image2"]).permute(2, 0, 1),
        }
        return image_data_point

    def __getitem__(self, index):
        if self._transform is None:
            raise AttributeError("Transform is None, you should apply set transform before")
        if self._right_video_dataset is not None and self._mono_video:
            image_data_point = self.stereo_video_item(index)
        elif self._right_video_dataset is not None:
            image_data_point = self.stereo_item(index)
        else:
            image_data_point = self.mono_video_item(index)
        if self._pose_dataset is None:
            return {**image_data_point}
        return {**image_data_point, **self._pose_dataset[index]}

    def __len__(self):
        return len(self._left_video_dataset) - 1
