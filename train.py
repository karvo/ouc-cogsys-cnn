#######################################################################################################################################
# Name: model.py
# Parent Code filename: transfer_learning_V6.py
# Author: Vasileios Karvonidis
# Creation date: 12.07.2022
# Last edit on: 12.07.2022
# Description: # Class for the creation of custom model for custom Convolutional Neural Network based on custom and unlabeled datasets
#
#   A basic flowchart of the code is the following:
#       1. Define the hyperparameters of the custom CNN (config.py)
#       2. Create and prepare the dataset in order to be model-compatible (dataset.py)
#       3. Create, configure and modify the model properly  (model.py)
#       4. Train the model <-------- You are here
#       5. Save the model <-------- You are also here
#       6. Create and save the model statistics in a report (metric.py)
#       7. Upload model metrics to tensorboard and create the appropriate url (main.py)
#######################################################################################################################################

import time
import torch
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import torch.nn as nn
from save import save_epoch, save_fold
from metrics import Metrics
import torch.optim as optim
from numpy.random import randn
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, confusion_matrix
from dataset import Dataset
from model import Model
from torch.utils.tensorboard import SummaryWriter

# Model training function
#
# Inputs: 
#          - model: the model to be trained
#          - dataloader: the dataloader
#          - dataset_size: the dataset size
#          - optimizer: the optimizer
#          - criterion: The loss function
# 
# Outputs:
#          - model: the trained model
#          - perfomance_recordings: the performance recordings

def train_epoch(model, dataloaders, criterion, optimizer, dataset_size, class_names, metrics, hp,epoch, saved_model_dir):

    for phase in ['train', 'test']:

        if phase == 'train':

            print("Training mode...")
            model.train()  # Set model to training mode

        else:

            print("Testing mode...")
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0   # Initialize Loss rate for current epoch
        running_corrects = 0 # Initialize correct prediction of current epoch

        all_preds = torch.tensor([]).to(hp['device'])  # Initialize empty list to store phase predictions
        all_labels = torch.tensor([]).to(hp['device']) # Initialize empty list to store phase predictions

        # Run the training batches
        for inputs, labels in tqdm(dataloaders[phase]):

            inputs = inputs.to(hp['device']) # Inputs are passed to device (GPU or CPU)
            labels = labels.to(hp['device']) # Outputs are passed to device (GPU or CPU)
        
            # If we are in train phase, enable gradient tracking, else disable it
            with torch.set_grad_enabled(phase == 'train'):

                outputs = model(inputs) # Pass the input batch to the model
                _, predictions = torch.max(outputs.data, 1)#Get the prediction in the form of 0/1
                loss = criterion(outputs, labels)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            # Update the statistics
            all_preds = torch.cat((all_preds, predictions))
            all_labels = torch.cat((all_labels, labels))
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predictions == labels.data)       
            
        epoch_loss = running_loss / dataset_size[phase]
        epoch_acc = running_corrects.double() / dataset_size[phase]
        print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')
  
        # Record the statistics of each phase
        metrics.get_metrics(all_labels,all_preds, class_names, epoch, epoch_acc, epoch_loss,phase,all_preds)

        if phase == 'test':
            save_epoch(model, saved_model_dir,hp['model_architecture'], str(epoch), hp['unique_id'] )
    
    return epoch_loss

def train(hp, model, dataset, dataset_size, class_names, optimizer, criterion, exp_lr_scheduler=None, estop=None):
    print("Starting training...")
    dataloaders = dataset.dataloader
    metrics = Metrics(hp) #Initialize metrics

    start_time = time.time() #Start timer

    # Initialize training parameter for keeping statistics
    model = model.to(hp['device'])
    estop_status = False

    for epoch in range(hp['epochs']):

        if estop_status:
            break

        print('=' * 50)
        print(f"Epoch {epoch}/{hp['epochs'] - 1}")
        print('=' * 50)
        
        epoch_loss = train_epoch(model, dataloaders, criterion, optimizer, dataset_size, class_names, metrics, hp,epoch,hp['saved_model_path'])
           
        if hp['learning_scheduler_on']:
            exp_lr_scheduler.step(epoch_loss)
        elif hp['early_stopping_on']:
            if estop(model, epoch_loss): estop_status = True
            print(f"Early stopping status: {estop.status}")

        # Collect the epoch performance
        metrics.collect_epoch_performance(epoch)

    metrics.gather_all_metrics(hp, dataset.dataset_info, str(model))

    total_time = time.time() - start_time
    print(f'Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
    print(f"Best val Acc: {metrics.performance['testing_accuracy_datapoints'].loc[metrics.best_epoch]}, in epoch: {metrics.best_epoch}")

def kfold_train(hp, dataset, dataset_size, class_names):
    
    kfold_dataset = ConcatDataset([dataset.train_data, dataset.test_data])


    k=10
    splits=KFold(n_splits=k,shuffle=True,random_state=42)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(kfold_dataset)))):
        model = Model(hp,dataset)
        model = model.to(hp['device'])

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(kfold_dataset, batch_size=hp['batch_size'], sampler=train_sampler)
        test_loader = DataLoader(kfold_dataset, batch_size=hp['batch_size'], sampler=test_sampler)
        dataloaders = {'train': train_loader, 'test': test_loader}
        train_size = len(train_sampler) # Training dataset size
        test_size = len(test_sampler) # Test dataset size
        
        dataset_size = {'train': train_size, 'test': test_size}
        start_time = time.time() #Start timer
        print('Fold {}'.format(fold + 1))
        model_dir = save_fold(hp['saved_model_path'],hp['model_architecture'], fold, hp['unique_id'])
        
        metrics = Metrics(hp, fold_path=model_dir.split('/')[-2]+ r'/')

        estop_status = False

        for epoch in range(hp['epochs']):

            print('=' * 50)
            print(f"Epoch {epoch}/{hp['epochs'] - 1}")
            print('=' * 50)
            
            if estop_status:
                break
        
            epoch_loss = train_epoch(model.model, dataloaders, model.criterion, model.optimizer, dataset_size, class_names, metrics, hp,epoch,model_dir)
        
            # Collect the epoch performance
            metrics.collect_epoch_performance(epoch)
        
            if hp['learning_scheduler_on']:
                model.exp_lr_scheduler.step(epoch_loss)
            elif hp['early_stopping_on']:
                if model.estop(model, epoch_loss): estop_status = True
                print(f"Early stopping status: {model.estop.status}")
            
        metrics.gather_all_metrics(hp, dataset.dataset_info, str(model))

    total_time = time.time() - start_time
    print(f'Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
    print(f"Best val Acc: {metrics.performance['testing_accuracy_datapoints'].loc[metrics.best_epoch]}, in epoch: {metrics.best_epoch}")