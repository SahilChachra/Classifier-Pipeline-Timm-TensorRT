import torch
import torch.nn as nn
import os
from dataset import CustomDatasetInf
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
from loguru import logger

def get_dataset_list(image_path):
    image_names = os.listdir(image_path)
    image_list = []
    for image_name in image_names:
        image_list.append(os.path.join(image_path, image_name))
    return image_list

@logger.catch()
def main(args):
    model_path = args.model_path
    image_path = args.image_path
    labels = args.labels
    batch_size = args.batch_size
    img_size = args.img_size
    device = args.device

    # Setup model
    model = torch.load(model_path)

    # Setup dataset
    data_list = get_dataset_list(image_path)
    
    customtransforms = {
        "test" : A.Compose([
                  A.Resize(img_size, img_size),
                  A.Normalize(p=1.0),
                  ToTensorV2(p=1.0)
            ])
        }

    test_images = CustomDatasetInf(data_list=data_list, transform=customtransforms["test"])
    test_loader = DataLoader(test_images, shuffle=True, batch_size=batch_size)

    # Setup labels
    label_temp = labels.split(",")
    labels = [i.strip() for i in label_temp]

    # inference
    y_pred_list = []
    image_path_list = []

    progress = tqdm(test_loader, desc="Validation", total=len(loader))

    for _, (images, img_path) in enumerate(progress):
    
        images = images.to(device)
        
        with torch.no_grad():
            model.eval()
            predictions = model(images)
        
        y_true = labels.long().squeeze()
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        top_prob, top_label = torch.topk(probabilities, 1)
        top_label = torch.flatten(top_label)
        
        y_pred_list = np.concatenate((y_pred_list, top_label.cpu().detach().numpy()), axis=0)
        image_path_list = np.concatenate((image_path_list, img_path), axis=0)

        torch.cuda.empty_cache()
        del predictions, images, img_path, loss
        gc.collect()

def arguement_parser():
    parser = argparse.ArgumentParser(description="Parse input for model training")

    parser.add_argument('--model_path', type=str, default="classifier.pt", help="PyTorch model path")
    parser.add_argument('--image_path', type=str, default="./images", help='Path to images')
    parser.add_argument('--labels', type=str, default="buildings, forests, mountains, glacier, street, sea", help='labels')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for inference')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--device', type=str, default="cuda", help='Device')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arguement_parser()
    main(args)
