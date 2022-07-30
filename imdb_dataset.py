import os

import PIL.Image
import pandas as pd
import torch
from matplotlib.pyplot import sca
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection

# scikit - train test div
class IMDBDataset(Dataset):
    def __init__(self, is_train=True):
        self.img_dir = "data"
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop((2304,2304)),
            transforms.Resize((128,128))
        ])
        images = os.listdir(self.img_dir)
        self.image_list = list()
        self.rating_list = list()

        for image in images:
            tokens = image.split("_")
            rating = float(tokens[0])
            self.image_list.append(image)
            self.rating_list.append(float(rating))

        self.__scale__()

        self.x_train, self.x_test,self.y_train, self.y_test = \
            model_selection.train_test_split(
                self.image_list, self.rating_list, test_size=0.2, random_state=11)

        if is_train:
            self.image_list = self.x_train
            self.rating_list = self.y_train
        else:
            self.image_list = self.x_test
            self.rating_list = self.y_test

    def __scale__(self):
        labels = [[i] for i in self.rating_list]
        self.scaler = MinMaxScaler()
        labels = self.scaler.fit_transform(labels)
        labels = torch.tensor(labels, dtype=torch.float32)
        self.rating_list = torch.squeeze(labels)

    def unscale(self, values):
        values = [[i] for i in values]
        values = self.scaler.inverse_transform(values)
        values = [i[0] for i in values]
        return values

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        rating = self.rating_list[idx]
        img_path = os.path.join(self.img_dir, image_name)
        image = PIL.Image.open(img_path)
        image = self.transforms(image)
        return image, rating

if __name__ == "__main__":
    cid = IMDBDataset(is_train=True)
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    print(len(cid))
    for image, label in dataloader:
        print(image.shape)
        print(label)
        exit(0)
        for i in image:
            plt.imshow(i[0].numpy())
            plt.show()
            exit(0)

