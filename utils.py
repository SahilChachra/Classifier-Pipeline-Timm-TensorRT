import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch import optim
from loguru import logger

dataset_path = ""

def set_datasetpath(path_):
    global dataset_path 
    dataset_path = path_

def get_image_path(image_name):
    global dataset_path
    return os.path.join(dataset_path, image_name)

def train_func(epoch, model, loader, device, optimizer, loss_function):
    
    model.train()

    y_pred_list = []
    y_true_list=[]
    running_loss = 0
    curr_num_of_data_read = 0
    # correct = 0

    progress = tqdm(enumerate(loader), desc="Training", total=len(loader))
    
    for _, (images, labels) in progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()*labels.shape[0]

        # THIS CODE WAS CONSIDERING INDEX AS LABEL (ARGMAX RETURNS INDEX NOT LABEL)
        '''y_pred, y_true = torch.argmax(predictions, axis=1), labels.long().squeeze()
        
        y_pred_list = np.concatenate((y_pred_list, y_pred.cpu().detach().numpy()), axis=0)
        y_true_list = np.concatenate((y_true_list, y_true.cpu().detach().numpy()), axis=0)'''

        y_true = labels.long().squeeze()
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        top_prob, top_label = torch.topk(probabilities, 1)
        top_label = torch.flatten(top_label)
        
        y_pred_list = np.concatenate((y_pred_list, top_label.cpu().detach().numpy()), axis=0)
        y_true_list = np.concatenate((y_true_list, y_true.cpu().detach().numpy()),axis=0)
        
        running_acc = accuracy_score(y_pred_list, y_true_list)
        #correct += (y_pred == y_true).type(torch.float).sum().item()

        curr_num_of_data_read += images.shape[0]
        
        # _train_accuracy = correct/curr_num_of_data_read

        progress.set_postfix(Epoch=epoch, Train_loss=running_loss/curr_num_of_data_read, Train_acc = running_acc, LR=optimizer.param_groups[0]['lr'])

        torch.cuda.empty_cache()
        del images, labels, loss, predictions
        gc.collect()
    
    epoch_accuracy = accuracy_score(y_pred_list, y_true_list)
    return model, optimizer, running_loss/curr_num_of_data_read, epoch_accuracy

def validation_func(epoch, model, loader, device, loss_function):
    
    running_loss = 0
    
    y_pred_list = []
    y_true_list = []

    progress = tqdm(loader, desc="Validation", total=len(loader))
    curr_num_of_data_read = 0
    _running_accuracy = 0.0

    for _, (images, labels) in enumerate(progress):

        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            model.eval()
            predictions = model(images)
        
        loss = loss_function(predictions, labels)

        running_loss += loss.item()*labels.shape[0]
        
        # THIS CODE WAS CONSIDERING INDEX AS LABEL (ARGMAX RETURNS INDEX NOT LABEL)
        '''y_pred, y_true = torch.argmax(predictions, axis=1), labels.long().squeeze()

        y_pred_list = np.concatenate((y_pred_list, y_pred.cpu().detach().numpy()), axis=0)
        y_true_list = np.concatenate((y_true_list, y_true.cpu().detach().numpy()), axis=0)'''

        y_true = labels.long().squeeze()
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        top_prob, top_label = torch.topk(probabilities, 1)
        top_label = torch.flatten(top_label)
        
        y_pred_list = np.concatenate((y_pred_list, top_label.cpu().detach().numpy()), axis=0)
        y_true_list = np.concatenate((y_true_list, y_true.cpu().detach().numpy()),axis=0)

        curr_num_of_data_read += images.shape[0]
        
        _running_accuracy = accuracy_score(y_pred_list, y_true_list)

        progress.set_postfix(Epoch=epoch, Val_loss=running_loss/curr_num_of_data_read, Val_accuracy =_running_accuracy)

        torch.cuda.empty_cache()
        del predictions, images, labels, loss
        gc.collect()
        
    accuracy = accuracy_score(y_pred_list, y_true_list)
    conf_matrix = confusion_matrix(y_true_list, y_pred_list)

    return running_loss/curr_num_of_data_read, accuracy, conf_matrix

def save_confusion_matrix(c_m, labels, exp_name, name):
    plt.rcParams['figure.figsize'] = (15.0, 15.0)
    plt.rcParams['font.size'] = 20

    # Implementing visualization of Confusion Matrix
    display_c_m = ConfusionMatrixDisplay(c_m, display_labels=labels)

    # Plotting Confusion Matrix
    # Setting colour map to be used
    display_c_m.plot(cmap='OrRd', xticks_rotation=25)
    # Other possible options for colour map are:
    # 'autumn_r', 'Blues', 'cool', 'Greens', 'Greys', 'PuRd', 'copper_r'

    # Setting fontsize for xticks and yticks
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Giving name to the plot
    plt.title(name, fontsize=24)

    # Saving plot
    plt.savefig(os.path.join(exp_name,name+'.png'), dpi=500)

def get_optimizer(optimizer_name, model, learning_rate):

    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    
    elif optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=learning_rate)
    
    elif optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate)
    
    else:
        logger.info("This optimizer is not yet present in the pipeline!")
        logger.info("Using Adam by default!")
        return optim.Adam(model.parameters(), lr=learning_rate)
    
def get_lrscheduler(scheduler_name, optimizer):
    if scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)

def get_loss_function(loss_func, target_size):
    if loss_func == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    else:
        logger.info("Unknown Loss function found! Using CrossEntropyLoss")
        return torch.nn.CrossEntropyLoss()

# Only for testing Utils.py!
if __name__ == "__main__":
    set_datasetpath("local")
    print(get_image_path("img.jpg"))