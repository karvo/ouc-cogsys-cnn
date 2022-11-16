#######################################################################################################################################
# Name: dataset.py
# Parent Code filename: transfer_learning_V6.py
# Author: Vasileios Karvonidis
# Creation date: 12.07.2022
# Last edit on: 12.07.2022
# Description: # Class for the creation of custom dataset for custom Convolutional Neural Network based on custom and unlabeled datasets
#
#   A basic flowchart of the code is the following:
#       1. Define the hyperparameters of the custom CNN (config.py)
#       2. Create and prepare the dataset in order to be model-compatible <-------- You are here
#       3. Create, configure and modify the model properly (model.py)
#       4. Train the model (train.py)
#       5. Save the model (train.py)
#       6. Create and save the model statistics in a report (metrics.py)
#       7. Upload model metrics to tensorboard and create the appropriate url (main.py)
#######################################################################################################################################

import os

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

class Dataset(Dataset):

    def __init__(self, hp):
        print("Creating the Dataset...")

        # Data directory root for training and testing
        self.root = hp['dataset_path'] 
        # Batch size of the dataset       
        self.batch_size = hp['batch_size'] 


        self.train_transform = transforms.Compose([
                        # Rotate +/- 10 degrees
                        #transforms.RandomRotation(150),      
                        #transforms.RandomAutocontrast(),
                        #transforms.RandomHorizontalFlip(),
                        # Reverse 50% of images
                        transforms.RandomVerticalFlip(),
                        # Resize shortest side to 224 pixels
                        transforms.Resize(256), 
                        # Crop longest side to 224 pixels at center            
                        transforms.CenterCrop(224),         
                        transforms.ToTensor(),
                        # Transforms.ColorJitter(contrast=1),
                        # Transforms.Grayscale(3),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225]),
                        transforms.RandomErasing()
                    ])

        self.test_transform = transforms.Compose([
                # Resize shortest side to 224 pixels
                transforms.Resize(256),
                # Crop longest side to 224 pixels at center             
                transforms.CenterCrop(224),       
                transforms.ToTensor(),       
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])

        # Load the images in dataset
        
        self.train_data = datasets.ImageFolder(os.path.join(self.root,'train'), transform = self.train_transform)
        self.test_data = datasets.ImageFolder(os.path.join(self.root,'test'), transform = self.test_transform)

        # Create the dataloaders
        
        train_loader = DataLoader(self.train_data, batch_size=hp['batch_size'], shuffle = True)
        test_loader = DataLoader(self.test_data, batch_size=hp['batch_size'], shuffle = True)
        self.dataloader = {'train': train_loader, 'test': test_loader}

        self.train_size = len(self.train_data) # Training dataset size
        self.test_size = len(self.test_data) # Test dataset size
        self.dataset_size = {'train': self.train_size, 'test': self.test_size}
        self.class_names = self.train_data.classes # Class names

        self.dataset_info = {
                # DATASET INFO
                'dataset_size' : self.dataset_size['train'] + self.dataset_size['test'],
                'training_dataset_size' : self.train_size,
                'testing_dataset_size' : self.test_size,
                'class_names' : self.class_names
        }

        print("================================================")
        print(f"Class names: {self.class_names}")
        print(f"Complete Dataset size: {self.dataset_size['train'] + self.dataset_size['test']}")
        print(f"Training Dataset size: {self.train_size}")
        print(f"Testing Dataset size: {self.test_size}")
        print("Dataset Ready!")
        print("================================================")