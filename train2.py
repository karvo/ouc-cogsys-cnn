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
#       6. Create and save the model statistics in a report
#       7. Upload model metrics to tensorboard and create the appropriate url              
#######################################################################################################################################


import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

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
# Ouputs:

def train(hp, model, dataloader, dataset_size, class_names, optimizer, criterion, exp_lr_scheduler):

    writer = SummaryWriter(hp['saved_model_path']) # Start Tensorboard to track metrics

    start_time = time.time() #Start timer

    # Initialize training parameter for keeping statistics
    model = model.to(hp['device'])
    train_size = dataset_size['train']
    test_size = dataset_size['test']

    train_losses = test_losses = train_correct = test_correct = []

    for epoch in range(hp['epochs']):
        
        print(f"Epoch {epoch}/{hp['epochs'] - 1}")
        print('-' * 10)
        
        train_corrects = 0
        test_corrects = 0

        # Run the training batches
        for b, (inputs, labels) in enumerate(dataloader['train']):

            if b == train_size:
                break
            b+=1

            inputs = inputs.to(hp['device']) # Inputs are passed to device (GPU or CPU)
            labels = labels.to(hp['device']) # Outputs are passed to device (GPU or CPU)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            predicted = torch.max(outputs.data, 1)[1]
            corrects = (predicted == labels).sum()
            train_corrects += corrects

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'epoch: {epoch:2}  batch: {b:4} [{10*b:6}/{train_size}]  loss: {loss.item():10.8f} accuracy: {train_corrects.item()*100/(10*b):7.3f}%')

            train_losses.append(loss)
            train_correct.append(train_corrects)

        # Run the testing batches
        with torch.no_grad():
            for b, (inputs, labels) in enumerate(dataloader['test']):
                if b == test_size:
                    break
                
                inputs = inputs.to(hp['device']) # Inputs are passed to device (GPU or CPU)
                labels = labels.to(hp['device']) # Outputs are passed to device (GPU or CPU)

                outputs = model(inputs)

                predicted = torch.max(outputs.data, 1)[1] 
                test_corrects += (predicted == labels).sum()

        loss = criterion(outputs, labels)
        test_losses.append(loss.cpu())
        test_correct.append(test_corrects.cpu())
    
    test_size = dataset_size['test']
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
    print(f'Test accuracy: {test_correct[-1].item()*100/test_size:.3f}%')

    torch.save(model.state_dict(), hp['saved_model_filename'])
    print(hp['saved_model_filename'])
    
