from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms as transforms
import albumentations


class ValidationDataset(Dataset):
    def __init__(self, main_folder='datasets/tum_3'):
        self.main_folder = main_folder
        self.id = 0
        with open(os.path.join(os.curdir, main_folder, "rgb.txt")) as f:
            self.rgb = f.read().splitlines()
            self.rgb = [el.split(" ")[1] for el in self.rgb]
            self.rgb = self.rgb[3:]
        with open(os.path.join(os.curdir, main_folder, "depth.txt")) as f:
            self.depth = f.read().splitlines()
            self.depth = [el.split(" ")[1] for el in self.depth]
            self.depth = self.depth[3:]
        self.length = len(self.depth)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # took form albumentations norm default values
            ]) 
    
    def __getitem__(self, id):
        path4depth = os.path.join(os.curdir, self.main_folder, self.depth[id])
        path4rgb = os.path.join(os.curdir, self.main_folder, self.rgb[id])
        img = cv2.imread(path4rgb)
        groundtruth_dict = {'image' : img, 'tensor': self.transform(img)[None], 'groundtruth_depth': 255 - cv2.imread(path4depth, 0)}
        return groundtruth_dict
    
    def __len__(self):
        return self.length
    
