import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def get_image_path(image):
    return os.path.join("/home/sahil/Documents/Classifiers/datasets/Plant_health_dataset", "images", image+".jpg")

def train_func(epoch, model, loader, device, optimizer, loss_function):
    
    model.train()

    running_loss = 0
    pred_for_acc = []
    labels_for_acc = []

    curr_num_of_data_read = 0

    progress = tqdm(enumerate(loader), desc="Training", total=len(loader))
    
    for _, (images, labels) in progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()*labels.shape[0]
        labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().detach().numpy()), axis=0)
        pred_for_acc = np.concatenate((pred_for_acc, np.argmax(predictions.cpu().detach().numpy(), axis=1)),axis=0)
        curr_num_of_data_read += images.shape[0]
        _train_accuracy = accuracy_score(labels_for_acc, pred_for_acc)

        progress.set_postfix(Epoch=epoch, Train_loss=running_loss/curr_num_of_data_read, Train_acc = _train_accuracy, LR=optimizer.param_groups[0]['lr'])

        torch.cuda.empty_cache()
        del images, labels, loss, predictions
        gc.collect()
        
    return model, optimizer, running_loss/curr_num_of_data_read, _train_accuracy

def validation_func(epoch, model, loader, device, loss_function):
    
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []
    
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
        labels_for_acc = np.concatenate((labels_for_acc,labels.cpu().detach().numpy()), axis=0)
        preds_for_acc = np.concatenate((preds_for_acc,np.argmax(predictions.cpu().detach().numpy(), axis=1)), axis=0)

        curr_num_of_data_read += images.shape[0]
        _running_accuracy = accuracy_score(labels_for_acc, preds_for_acc)

        progress.set_postfix(Epoch=epoch, Val_loss=running_loss/curr_num_of_data_read, Val_accuracy =_running_accuracy)

        torch.cuda.empty_cache()
        del predictions, images, labels, loss
        gc.collect()
        
    accuracy = accuracy_score(labels_for_acc, preds_for_acc)
    conf_matrix = confusion_matrix(labels_for_acc, preds_for_acc)

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
