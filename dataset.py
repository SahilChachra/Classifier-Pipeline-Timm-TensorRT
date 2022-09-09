from PIL import Image
import torch
import os
from skimage import io
import pandas as pd
from torch.utils.data import Dataset

# For Training the model
class CustomDataset(Dataset):
    def __init__(self, data_csv, data_labels, root_dir, test=False, transform=None):
        """
        Parameters :-

        data_csv : Pandas DataFrame with image_path as columns
        data_labels : Pandas DataFrame with labels as columns
        root_dir : Dataset root directory
        test : Are we testing the model?
        transform : Transforms
        """
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

# For inferencing the model independently using Torch
class CustomDatasetInf(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Parameters :-
        
        data_list : List having image paths
        transforms : transforms for testing
        """
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