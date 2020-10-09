import os

import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# noinspection PyUnresolvedReferences
class TumValidationDataset(Dataset):
    def __init__(self, main_folder='datasets/tum_3', final_img_size=(128, 384)):
        self._main_folder = main_folder
        with open(os.path.join(main_folder, "rgb.txt")) as fd:
            self._rgb_images = fd.read().splitlines()
            self._rgb_images = [element.split(" ")[1] for element in self._rgb_images]
            self._rgb_images = self._rgb_images[3:]
        with open(os.path.join(main_folder, "depth.txt")) as fd:
            self._depth_images = fd.read().splitlines()
            self._depth_images = [element.split(" ")[1] for element in self._depth_images]
            self._depth_images = self._depth_images[3:]
        self._length = len(self._depth_images)
        self._final_img_size = final_img_size

        image = cv2.imread(os.path.join(self._main_folder, self._rgb_images[0]))
        used_img_size = image.shape[:2]
        self._ratio = max(float(final_img_size[0]) / used_img_size[0],
                          float(final_img_size[0]) / used_img_size[0])
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((used_img_size[0] * self._ratio, used_img_size[1] * self._ratio)),
            transforms.CenterCrop(final_img_size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        path4depth = os.path.join(self._main_folder, self._depth_images[index])
        path4rgb = os.path.join(self._main_folder, self._rgb_images[index])
        image = cv2.imread(path4rgb)
        depth = 255 - cv2.imread(path4depth, 0)
        ground_truth_dict = {'image': self._transform(image)[None],
                             'ground_truth_depth': self._transform_depth(depth)}
        return ground_truth_dict

    def _transform_depth(self, depth):
        used_img_size = depth.shape[:2]
        depth = cv2.resize(depth, (used_img_size[0] * self._ratio, used_img_size[1] * self._ratio))
        height = int(used_img_size[0] * self._ratio)
        width = int(used_img_size[1] * self._ratio)
        delta_height = int((used_img_size[0] - height) / 2.)
        delta_width = int((used_img_size[1] - width) / 2.)
        depth = depth[delta_height:delta_height + height, delta_width:delta_width + width]
        return cv2.resize(depth, self._final_img_size)

    def __len__(self):
        return self._length
