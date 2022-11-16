#######################################################################################################################################
# Name: config.py
# Parent Code filename: transfer_learning_V6.py
# Author: Vasileios Karvonidis
# Creation date: 12.07.2022
# Last edit on: 12.07.2022
#
# Description: 
# 
#  Class for the creation of a dictionary which contains all the 
#  necessary hyperparameters for cthe ustom Convolutional Neural Network based 
#  on custom and unlabeled datasets.
#
#   A basic flowchart of the code is the following:
#       1. Define the hyperparameters of the custom CNN <-------- You are here
#       2. Create and prepare the dataset in order to be model-compatible (dataset.py)
#       3. Create, configure and modify the model properly (model.py)
#       4. Train the model (train.py)
#       5. Save the model (train.py)
#       6. Create and save the model statistics in a report (metrics.py)
#       7. Upload model metrics to tensorboard and create the appropriate url (main.py)
#######################################################################################################################################

import time
import torch
from torchvision import transforms

class Hyperparameters():
    # In case more hyperparameters are added, make sure to add them in the dictionary
    def __init__(self):
        print("Configuring the Hyperparameters...")
        #MODEL PARAMETERS
        model_architecture_lst = ['vgg19', 'resnet50', 'custom']
        self.model_architecture = model_architecture_lst[0] # Define the model architecture
        self.pretrained_model = True # Define if you want the custom model architecture to be pre-trained. For this application this is set True always
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Define the device for calculation. If GPU is available, model is trained there, else in CPU

        # DIRECTORIES
        self.unique_id = time.strftime("%Y%m%d_%H%M%S") #Unique ID for the new model
        self.model_name =  self.model_architecture + "_" + self.unique_id
        saved_model_root_path = r'/media/kv/Documents/git/mtkvcs-saved-models/'
        self.saved_model_path = saved_model_root_path + self.model_name + r'/' # Create new path for the model
        self.saved_model_filename = self.saved_model_path + self.model_architecture + r'_' + time.strftime("%Y%m%d_%H%M%S") + r'.pt' # Define filename model based on the time of creation
        self.root = r'/media/kv/Documents/git/mtkvcs-dataset/datscan/' #Define the dataset path

        # DATASET PARAMETERS
        self.batch_size = 64 # Batch size of the dataloader
        self.train_dataset_shuffle = True # Define if the train data will be shuffle. This is set always to True

        #OPTIMIZER PARAMETERS
        self.optimizer_function = 'Adam Optimizer' # Select the optimizer
        self.learning_rate = 0.0001 #Define the learning rate

        #LOSS FUNCTION PARAMETERS
        self.loss_function_type = 'Cross Entropy Loss Function' # Define the loss function

        # LEARNING SCHEDULER PARAMETERS
        self.early_stopping_on = False
        self.steplr_on = not self.early_stopping_on
        lr_list = ['Step Learning Rate', 'Reduce LR On Plateau']
        self.learning_scheduler_type = lr_list[1] # Define the learning rate scheduler
        self.gamma = 0.1
        self.learning_scheduler_step_size = 3

        # TRAINING PARAMETERS
        self.epochs = 15

        self.hp = {
                # DIRECTORIES
                '\nMODEL DIRECTORIES' : "",
                'unique_id' : self.unique_id,
                'model_name': self.model_name,
                'saved_model_path' : self.saved_model_path,
                'saved_model_filename' : self.saved_model_filename,
                'dataset_path' : self.root,

                # MODEL PARAMETERS
                '\nMODEL PARAMETERS' : "",
                'model_architecture' : self.model_architecture,
                'pretrained_model' : self.pretrained_model,
                'device': self.device,

                #OPTIMIZER PARAMETERS
                '\nOPTIMIZER PARAMETERS' : "",
                'optimizer_function' : self.optimizer_function,    
                'learning_rate' : self.learning_rate,

                #LOSS FUNCTION PARAMETERS
                '\nLOSS FUNCTION PARAMETERS' : "",
                'loss_function_type': self.loss_function_type,

                # LEARNING SCHEDULER PARAMETERS
                         
                '\nLEARNING SCHEDULER PARAMETERS' : "",
                'early_stopping_on' : self.early_stopping_on,
                'learning_scheduler_on' : self.steplr_on,
                'learning_scheduler_type' : self.learning_scheduler_type,
                'gamma' : self.gamma,
                'learning_scheduler_step_size': self.learning_scheduler_step_size,

                #TRAINING PARAMETERS
                '\nTRAINING PARAMETERS' : "",
                'epochs' : self.epochs,

                #DATASET PARAMETERS
                '\nDATASET PARAMETERS' : "",
                'batch_size' : self.batch_size,
                'dataset_shuffle': self.train_dataset_shuffle
            }
        print("Hyperparameters have been setup!")