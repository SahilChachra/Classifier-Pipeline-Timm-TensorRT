import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images_path, labels = None, test=False, transform=None):
        self.images_path = images_path
        self.test = test
        if self.test == False:
            self.labels = labels
            
        self.images_transform = transform

    def __getitem__(self, index):
        if self.test == False:
            labels = torch.tensor(self.labels.iloc[index])
  
        image = cv2.imread(self.images_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        image_transformed = self.images_transform(image=image)
        
        if self.test == False:
            return image_transformed["image"], labels
        return image_transformed["image"]

    def __len__(self):
        return self.images_path.shape[0]