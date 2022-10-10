from ast import arg
import os
import time
import argparse
from sched import scheduler
from tqdm import tqdm
import gc 
import shutil
from skimage import io
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from loguru import logger

from dataset import CustomDataset
from model import CustomModel
from utils import get_image_path, train_func, validation_func, save_confusion_matrix

def prepare_dataset(root_dataset, train_csv_path, split):
    train_data = pd.read_csv(train_csv_path)

    train_data["image_path"] = train_data["image_name"].apply(get_image_path)

    train_labels = train_data["label"]

    train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_data["image_path"], train_labels, test_size=split, random_state=22, stratify=train_labels)

    train_paths.reset_index(drop=True, inplace=True)
    train_labels.reset_index(drop=True, inplace=True)
    valid_paths.reset_index(drop=True, inplace=True)
    valid_labels.reset_index(drop=True, inplace=True)

    return train_paths, train_labels, valid_paths, valid_labels

@logger.catch
def main(args):

    # Set arguments
    model_name = args.model_name
    epochs = args.epochs
    batch_size = args.batch_size
    img_size = args.img_size
    device = args.device
    optimizer_name = args.optimizer
    learning_rate = args.lr
    lr_scheduler = args.lr_scheduler
    root_dataset = args.dataset
    split = args.split
    target_size = args.target_size
    early_stop = args.early_stop
    loss_func = args.loss_func
    exp_name = args.exp_name
    labels = args.labels
    wandb = args.wandb
    project_name = args.project_name
    # image_folder = args.image_folder

    if wandb:
        try:
            import wandb
        except:
            print("Wandb not installed. Running experiment without it.")
            wandb = False
        
        wandb.login()

        CONFIG = dict (
            num_labels = target_size,
            train_val_split = split,
            img_width = img_size,
            img_height = img_size,
            batch_size = batch_size,
            epochs = epochs,
            learning_rate = learning_rate,
            architecture = "CNN",
            infra = "Local",
            model_name = model_name,
            optimizer_name = optimizer_name
        )
    
        run = wandb.init(project=project_name, 
                 config=CONFIG, group=model_name, job_type="train")

    save_checkpoint_folder = os.path.join(exp_name, args.save_checkpoint_folder)
    save_model_folder = os.path.join(exp_name,args.save_model_folder)
    
    
    # Display input parameters
    logger.info(f"Training Settings : Epochs = {epochs}, \
                    batch_size = {batch_size}, img_size = {img_size}, \
                    optimizer_name = {optimizer_name}, learning_rate = {learning_rate}, \
                    lr_scheduler = {lr_scheduler}, split ratio = {split}, loss_func = {loss_func}, \
                    save_checkpoint_folder = {save_checkpoint_folder}, save_model_folder = {save_model_folder}, \
                    exp_name = {exp_name}")

    # Setup folders to save model, checkpoints and confusion matrix
    if not os.path.exists(exp_name):
        os.mkdir(exp_name)

    if not os.path.exists(save_checkpoint_folder):
        os.mkdir(save_checkpoint_folder)
    
    if not os.path.exists(save_model_folder):
        os.mkdir(save_model_folder)

    train_csv_path = os.path.join(root_dataset, "train.csv")
    # images = os.path.join(root_dataset, image_folder)

    # Setup Device to train
    if "cuda" in device and torch.cuda.is_available():
        logger.info("Using GPU/CUDA")
    else:
        device = "cpu"
        logger.info("GPU/CUDA unavailable. Using CPU")   
    
    # Create dataset
    train_paths, train_labels, valid_paths, valid_labels = prepare_dataset(root_dataset, train_csv_path, split)
    
    # Setting up transforms
    customtransforms = {
        "train": A.Compose([
            #A.ToPILImage(),
            A.Resize(img_size, img_size),
            A.Flip(), 
            A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
            A.Normalize(p=1),
            ToTensorV2(p=1.0)
            ]
        ),
        "valid" : A.Compose([
                  #A.ToPILImage(),
                  A.Resize(img_size, img_size),
                  A.Normalize(p=1.0),
                  ToTensorV2(p=1.0)
            ])
        }
    
    # Setting up dataloaders
    train_images = CustomDataset(data_csv=train_paths, data_labels=train_labels, root_dir=root_dataset, test=False, transform=customtransforms["train"])
    train_loader = DataLoader(train_images, shuffle=True, batch_size=batch_size)

    valid_images = CustomDataset(data_csv=valid_paths, data_labels=valid_labels, root_dir=root_dataset, test=False, transform=customtransforms["valid"])
    valid_loader = DataLoader(valid_images, shuffle=True, batch_size=batch_size)

    # Set up the model, loss function and optimizers
    model = CustomModel(model_name=model_name, target_size=target_size, pretrained=True)
    model.to(device)

    # Set up variable to torch.jit.script
    m = torch.jit.script(CustomModel(model_name=model_name, target_size=target_size, pretrained=True))

    # Set up loss function
    loss_function_dict = {
                    "CrossEntropyLoss" : nn.CrossEntropyLoss(), 
                }
    
    loss_function = loss_function_dict[loss_func]

    # Set up Optimizer
    optimizer_dict = {
        "Adam" : optim.Adam(model.parameters(), lr=learning_rate),
        "AdamW" : optim.AdamW(model.parameters(), lr=learning_rate),
        "SGD" : optim.SGD(model.parameters(), lr=learning_rate)
        }
    optimizer = optimizer_dict[optimizer_name]

    # Set up LR Scheduler
    if optimizer_name not in ["SGD"] and lr_scheduler == "LambdaLR":
        logger.info(f"Only SGD supports LambdaLR Scheduler. Setting Scheduler to None or restart training with correct params.")
        lr_scheduler = None

    if lr_scheduler:
        scheduler_dict = {
            "CosineAnnealingLR" : torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0),
            "CosineAnnealingWarmRestarts" : torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)
            }
        if optimizer_name == "SGD" and lr_scheduler == "LambdaLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=learning_rate*100,step_size_up=5,mode="triangular2"),
        else:
            scheduler = scheduler_dict[lr_scheduler]
        del scheduler_dict
    
    del optimizer_dict, loss_function_dict
    gc.collect()

    # Set up training variables
    train_loss = []
    valid_loss = []
    train_acc = []
    val_acc = []
    lrs = []

    # Create labels list for confusion matrix
    label_temp = labels.split(",")
    labels = [i.strip() for i in label_temp]
    
    if len(labels) != target_size:
        print("Number of labels and target size do not match!")
        exit(0)

    # Initialize some extra variables
    max_val_acc = -np.inf # Initialize maximum accuracy with -infinity
    no_val_acc_improve = 0
    best_model = model

    start_time = time.time() # Timer for overall training time
    
    if wandb:
        wandb.watch(model, loss_function, log="all", log_freq=5)

    # Start Training
    for epoch in range(1,epochs+1):
        epoch_start_time = time.time() # Timer for single epoch
        logger.info(f"Epoch : {epoch}")

        model, optimizer, train_loss_epoch, train_accuracy_epoch = train_func(epoch, model, train_loader, device, optimizer, loss_function)

        train_loss.append(train_loss_epoch)
        train_acc.append(train_accuracy_epoch)
        lrs.append(optimizer.param_groups[0]["lr"])

        valid_loss_epoch, valid_accuracy_epoch, conf_matrix = validation_func(epoch, model, valid_loader, device, loss_function)

        valid_loss.append(valid_loss_epoch)
        val_acc.append(valid_accuracy_epoch)

        if valid_accuracy_epoch > max_val_acc:
            logger.info(f"Accuracy improved from {max_val_acc:.3f} to {valid_accuracy_epoch:.3f}!")
            state = {"epoch":epoch, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), "loss":train_loss_epoch}
            torch.save(state, os.path.join(save_checkpoint_folder,f"classifier_statedict_ep{epoch}_{valid_accuracy_epoch:.3f}.pt"))
            #torch.save(model, os.path.join(save_model_folder,f"classifier_ep{epoch}_{valid_accuracy_epoch:.3f}.pt"))
            m.save(f"classifier_ep{epoch}_{valid_accuracy_epoch:.3f}.pt") # Saves model as .pt using torch.jit.save
            max_val_acc = valid_accuracy_epoch
            no_val_acc_improve = 0
            best_model = model
            del state
            gc.collect()

        else:
            no_val_acc_improve += 1
        
        if lr_scheduler is not None:
            scheduler.step()

        epoch_end_time = time.time() # Timer for single epoch
        epoch_time = epoch_end_time-epoch_start_time

        info = f"Epoch : {epoch}, Train loss : {train_loss_epoch:.3f}, Valid loss : {valid_loss_epoch:.3f}, Train acc : {train_accuracy_epoch:.3f}, Val acc : {valid_accuracy_epoch:.3f}, Time taken : {epoch_time/60:.3f} mins"
        tqdm.write(info)

        if wandb:
            wandb.log({"Epoch":epoch, "Train_loss" : train_loss_epoch, "Valid_loss" : valid_loss_epoch, 
                  "Train_acc" : train_accuracy_epoch, "Val_acc" : valid_accuracy_epoch})

        torch.cuda.empty_cache()
        gc.collect()

        # Early Stopping
        if no_val_acc_improve == early_stop:
            logger.info(f"Validation accuracy did not approve for upto {no_val_acc_improve} epoch(s). Stopping training...")
            logger.info(f"Saving State dictionary")
            state = {"epoch":epoch, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), "loss":train_loss_epoch}
            torch.save(state, os.path.join(save_checkpoint_folder,f"classifier_statedict_{epoch}.pt"))
            break
        
        # For last epoch
        if epoch == epochs+1:
            logger.info(f"Training job is ending. Saving final epoch's model state diectionary.")
            state = {"epoch":epoch, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), "loss":train_loss_epoch}
            torch.save(state, os.path.join(save_checkpoint_folder,f"classifier_statedict_{epoch}.pt"))
    
    end_time = time.time() # Timer for overall training time
    total_training_time = end_time - start_time
    print(f"Total Training time : {total_training_time/60:.3f} minutes")

    # Generate confusion matrix for Train and Val set
    logger.info(f"Generating Confusion matrix for Train and Validation set using the best model")

    train_loss_epoch, train_accuracy_epoch, train_conf_matrix = validation_func(epochs+1, best_model, train_loader, device, loss_function)
    save_confusion_matrix(train_conf_matrix, labels, exp_name, name="Train_ConfusionMatrix")

    valid_loss_epoch, valid_accuracy_epoch, valid_conf_matrix = validation_func(epochs+1, best_model, valid_loader, device, loss_function)
    save_confusion_matrix(valid_conf_matrix, labels, exp_name, name="Valid_ConfusionMatrix")

    if wandb:
        wandb.log({"Best accuracy on Train set":train_accuracy_epoch, "Best accuracy on Val set":valid_accuracy_epoch})
        wandb.log({"Train Confusion Matrix": wandb.Image(f"{exp_name}/Train_ConfusionMatrix.png")})
        wandb.log({"Valid Confusion Matrix": wandb.Image(f"{exp_name}/Valid_ConfusionMatrix.png")})

    del best_model
    gc.collect()

    if wandb:
        run.finish()

def arguement_parser():
    parser = argparse.ArgumentParser(description="Parse input for model training")

    parser.add_argument('--model_name', type=str, default="resnet50", help="Model name from Timm")
    parser.add_argument('--epochs', type=int, default=5, help='Total num of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size of training/validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='image size in pixels')
    parser.add_argument('--device', default='cuda', help='Device for training')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'AdamW', 'SGD'], default='Adam', help='Optimizer')
    parser.add_argument('--lr_scheduler', type=str, default="CosineAnnealingLR", choices=["CosineAnnealingLR", "LamdaLR", "CosineAnnealingWarmRestarts", None], help='Select learning rate scheduler')
    parser.add_argument('--dataset', type=str, default="/home/sahil/Documents/Classifiers/datasets/Plant_health_dataset", help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.20, help='Validation data split ratio')
    parser.add_argument('--target_size', type=int, default=6, help='Number of classes')
    parser.add_argument('--early_stop', type=int, default=5, help='Early stop threshold')
    parser.add_argument('--loss_func', type=str, default="CrossEntropyLoss", choices=["CrossEntropyLoss", "FocalLoss"], help="Select loss function")
    parser.add_argument('--save_checkpoint_folder', type=str, default="checkpoint", help="Save model checkpoint folder")
    parser.add_argument('--save_model_folder', type=str, default="weights", help="Save weight file folder")
    parser.add_argument('--exp_name', type=str, default="exp1", help="Save experiment data")
    parser.add_argument('--labels', type=str, default="buildings, forests, mountains, glacier, street, sea", help="Comma seperated labels")
    parser.add_argument('--wandb', action='store_true', help='Use W&B for MLOps')
    parser.add_argument('--project_name', type=str, default="Classifier-Pipeline", help="Name of project")
    # parser.add_argument('--image_folder', type=str, default="train", help="Images folder name")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arguement_parser()
    main(args)