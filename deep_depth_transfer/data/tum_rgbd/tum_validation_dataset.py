import os

import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

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
                          float(final_img_size[1]) / used_img_size[1])
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(used_img_size[0] * self._ratio), int(used_img_size[1] * self._ratio))),
            transforms.CenterCrop(final_img_size),
            transforms.ToTensor(),
        ])
        self._transform_depth = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((int(final_img_size[0] / self._ratio), int(final_img_size[1] / self._ratio))),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        path4depth = os.path.join(self._main_folder, self._depth_images[index])
        path4rgb = os.path.join(self._main_folder, self._rgb_images[index])
        image = cv2.imread(path4rgb)
        depth = cv2.imread(path4depth, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = depth / 5000.
        ground_truth_dict = {'image': self._transform(image),
                             'ground_truth_depth': self._transform_depth(depth)}
        return ground_truth_dict

    def __len__(self):
        return self._length
