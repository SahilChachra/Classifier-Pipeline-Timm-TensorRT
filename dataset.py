from PIL import Image
import torch
import os
from skimage import io
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

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

# Custom Image folder class. When using this, Comment out ToTensorV2
class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, total_classes=None):
        self.transform = transform
        self.data = []
        
        if total_classes:
            self.classnames  = os.listdir(root_dir)[:total_classes] # for test
        else:
            self.classnames = os.listdir(root_dir)
            
        for index, label in enumerate(self.classnames):
            root_image_name = os.path.join(root_dir, label)
            
            for i in os.listdir(root_image_name):
                full_path = os.path.join(root_image_name, i)
                self.data.append((full_path, index))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, target = self.data[index]
        img = np.array(Image.open(data))
        
        if self.transform:
            augmentations = self.transform(image=img)
            img = augmentations["image"]
        
        target = torch.from_numpy(np.array(target))
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        #print(type(img),img.shape, target)
        
        return img,target 

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
