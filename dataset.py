from PIL import Image
import torch
import os
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_csv, root_dir, transform=None):
        self.data_csv = pd.read_csv(data_csv)
        self.root_dir = root_dir
        self.transform = transform
                    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data_csv.iloc[index, "image_path"])
        image = Image.open(img_path)
        label = torch.tensor(int(self.data_csv.iloc[index, "label"]))
        if self.transform:
            image = self.transform(image)
        return [image, label]

