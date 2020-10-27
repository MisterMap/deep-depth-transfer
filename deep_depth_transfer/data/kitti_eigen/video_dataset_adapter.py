import os
import cv2
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class VideoDatasetAdapter(object):
    def __init__(self, main_folder, split, side):
        with open(os.path.join(main_folder, "splits", split, "train_files.txt")) as fd:
            self.filenames = fd.readlines()
        self._main_folder = main_folder
        self._length = len(self.filenames)
        self.loader = pil_loader
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self._img_size = [370, 1226]
        self._img_ext = ".png"
        self._side = side

    def get_image_size(self):
        return self._img_size

    def __getitem__(self, index):
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
        f_str = "{:010d}{}".format(frame_index, self._img_ext)
        image_path = os.path.join(
            self._main_folder, "kitti_data", folder, "image_0{}/data/{}".format(self.side_map[self._side], f_str))
        return self.loader(image_path)


    def __len__(self):
        return self._length