import os
import time
import torch

def save_epoch(model, folder, architecture, epoch, unique_id):
    path = architecture + r'_' + unique_id + "_epoch_" + epoch
    os.makedirs(folder+path)
    torch.save(model.state_dict(), folder + path + r'/'+ path + r'.pt')

def save_fold(folder, architecture, fold, unique_id):
    path = architecture + r'_' + unique_id + "_fold_" + str(fold) +r'/'
    os.makedirs(folder+path)
    
    return folder+path
    #torch.save(model.state_dict(), folder + path + r'/'+ path + r'.pt')

# Function to save the perfomance recordings to a plot
#
# Inputs: 
#          - dataframe: The dataframe
#          - tags: the tags of the plot to be created (label)
#
# Product: - Plot charts in the form of png
#
# Outputs: - None