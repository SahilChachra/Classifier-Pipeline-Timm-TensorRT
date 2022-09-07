from PIL import Image
import torch
import os
from skimage import io
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_csv, data_labels, root_dir, test=False, transform=None):
        self.data_csv = data_csv
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        if not self.test:
            self.data_labels = data_labels
                    
    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self, index):
        img_path = self.data_csv.iloc[index]
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image=image)["image"]
        if not self.test:
            label = torch.tensor(int(self.data_labels.iloc[index]))
            return [image, label]
        else:
            return [image]

class CustomDatasetInf(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
                    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        img_path = self.data_list[index]
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image=image)["image"]
        return [image, img_path]