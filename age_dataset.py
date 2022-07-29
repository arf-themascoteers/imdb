import os

import PIL.Image
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import os
import random

class AgeDataset(Dataset):
    def __init__(self, is_train=True):
        self.img_dir = "data/train"
        if is_train is False:
            self.img_dir = "data/test"
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop((2304,2304)),
            # transforms.Resize(128)
        ])
        ages = os.listdir(self.img_dir)
        self.image_list = {}
        self.age_list = {}
        i = 0
        for age in ages:
            images = os.listdir(self.img_dir+"/"+age)
            for image in images:
                self.image_list[i] = image
                self.age_list[i] = age
                i = i + 1

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        age = self.age_list[idx]
        img_path = os.path.join(self.img_dir, age, image_name)
        image = PIL.Image.open(img_path)
        image = self.transforms(image)
        label = float(age)
        return image, torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    cid = AgeDataset()
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    
    for image, label in dataloader:
        print(image.shape)
        print(label)
        for i in image:
            plt.imshow(i[0].numpy())
            plt.show()
            exit(0)

