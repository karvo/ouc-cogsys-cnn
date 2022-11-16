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
#       3. Create, configure and modify the model properly <-------- You are here
#       4. Train the model (train.py)
#       5. Save the model (train.py)
#       6. Create and save the model statistics in a report
#       7. Upload model metrics to tensorboard and create the appropriate url              
#######################################################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.optim import lr_scheduler
from early_stopping import EarlyStopping
#from learning_rate_scheduler import LRScheduler

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

class Model(nn.Module):

    def __init__(self, hp, dataset):
        super().__init__()
        self.hp = hp
        self.class_names = dataset.class_names

        if self.hp['model_architecture'] == 'vgg19':
            self.model = models.vgg19(pretrained=self.hp['pretrained_model']) # Initialize the model


            for self.param in self.model.parameters():
                self.param.requires_grad = False
            
            self.model.classifier = nn.Sequential(nn.Linear(25088,3136), # Configure the classifier
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(3136,3136, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(3136,len(self.class_names), bias=True))
            
            self.model_parameters = self.model.classifier.parameters()
            # note: Newly constructed layer has requires_grad=True by default. You don’t need to do it manually.

        elif self.hp['model_architecture']=='resnet50':

            self.model = models.resnet50(pretrained=self.hp['pretrained_model'])
            
            for self.param in self.model.parameters():
                self.param.requires_grad = False

            self.model.fc = nn.Linear(2048,len(self.class_names), bias=True)

            self.model_parameters = self.model.fc.parameters()
            # note: Newly constructed layer has requires_grad=True by default. You don’t need to do it manually.

        elif self.hp['model_architecture']=='custom':
            self.model = CustomCNN()
            self.model_parameters = self.model.parameters()
                            

        #print("Model architecture:\n\n") # Print the current model architecture
        #print(self.model)

        print(f'Current model exists in {"GPU" if torch.cuda.is_available() else "CPU"}.')
        self.model = self.model.to(self.hp['device']) #Pass model to GPU

        self.criterion = nn.CrossEntropyLoss() # Define loss function
        self.optimizer = optim.Adam(self.model_parameters, lr=self.hp['learning_rate']) #Define the optimizer

        if hp['learning_scheduler_on']:
            if hp['learning_scheduler_type'] == 'Step Learning Rate':
                self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.1)             
                #self.exp_lr_scheduler = LRScheduler(self.optimizer)
            elif hp['learning_scheduler_type'] == 'Reduce LR On Plateau':
                self.exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)             
        else:
            self.exp_lr_scheduler = None

        if hp['early_stopping_on']:
            self.estop = EarlyStopping()
        else:
            self.estop = None

        print("Optimizer: Adam *HARDCODED*")
        print("Loss function: Cross Entropy *HARDCODED*")
    