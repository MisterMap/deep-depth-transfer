import os
import cv2


class VideoDatasetAdapter(object):
    def __init__(self, main_folder):
        with open(os.path.join(main_folder, "rgb.txt")) as fd:
            self._rgb_images = fd.read().splitlines()
            self._rgb_images = [element.split(" ")[1] for element in self._rgb_images]
            self._rgb_images = self._rgb_images[3:]
        self._main_folder = main_folder
        image = cv2.imread(os.path.join(main_folder, self._rgb_images[0]))
        self._img_size = image.shape[:2]
        self._length = len(self._rgb_images)

    def get_image_size(self):
        return self._img_size

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self._main_folder, self._rgb_images[index]))
        return image

    def __len__(self):
        return self._length
