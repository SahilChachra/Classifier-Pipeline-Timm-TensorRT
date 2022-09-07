import torch
import torch.nn as nn
from dataset import CustomDatasetInf
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

def get_dataset_list(image_path):
    image_names = os.listdir(image_path)
    image_list = []
    for image_name in image_names:
        image_list.append(os.path.join(image_path, image_name))
    return image_list

def main(args):
    model_path = args.model_path
    image_path = args.image_path
    labels = args.labels
    batch_size = args.batch_size
    img_size = args.img_size

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


def arguement_parser():
    parser = argparse.ArgumentParser(description="Parse input for model training")

    parser.add_argument('--model_path', type=str, default="classifier.pt", help="PyTorch model path")
    parser.add_argument('--image_path', type=str, default="./images", help='Path to images')
    parser.add_argument('--labels', type=str, default="buildings, forests, mountains, glacier, street, sea", help='labels')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for inference')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arguement_parser()
    main(args)
