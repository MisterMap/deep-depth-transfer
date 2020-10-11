import os
from PIL import Image
import numpy as np


class VideoDatasetAdapter(object):
    def __init__(self, directory):
        self._image_names = sorted([os.path.join(directory, x) for x in os.listdir(directory)])

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, index):
        image = np.array(Image.open(self._image_names[index]))
        return image

    def get_image_size(self):
        return np.array(Image.open(self._image_names[0])).shape[::]