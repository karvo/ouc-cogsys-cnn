#######################################################################################################################################
# Name: main.py
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
#       4. Train the model (train.py)
#       5. Save the model (train.py)
#       6. Save the model statistics in a report (metrics.py)
#       7. Upload model metrics to tensorboard and create the appropriate url   <----- You are here       
#######################################################################################################################################

import os
from train import save_epoch
from train import train, kfold_train
#from save import save_model
from metrics import create_report
from model import Model
from dataset import Dataset
from config import Hyperparameters
import tensorboard

def main():

    # 1. Define the hyperparameters of the custom CNN (config.py)
    config_setup = Hyperparameters()

    # 2. Create and prepare the dataseft in order to be model-compatible (dataset.py)
    dataset = Dataset(config_setup.hp)

    # 3. Create, configure and modify the model properly  (model.py)
    model = Model(config_setup.hp, dataset)

    # 4. Train the model and record the statistics (train.py)
    train(config_setup.hp, model.model, dataset, dataset.dataset_size, dataset.class_names, model.optimizer, model.criterion, model.exp_lr_scheduler, model.estop)
    #kfold_train(config_setup.hp, dataset, dataset.dataset_size, dataset.class_names)
    
    # 5. Upload model metrics to tensorboard and create the appropriate url
    #os.system("tensorboard dev upload --logdir \ " + config_setup.hp['saved_model_path'])

if __name__ == "__main__":
    main()