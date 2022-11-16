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
#       4. Train the model (train.py)
#       5. Save the model (train.py)
#       6. Create and save the model statistics in a report <------ you are here
#       7. Upload model metrics to tensorboard and create the appropriate url (main.py)           
#######################################################################################################################################

import numpy as np
import pandas as pd
import seaborn as sn
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import f1_score, confusion_matrix

from torch.utils.tensorboard import SummaryWriter

class Metrics():

    def __init__(self, hp, fold_path = ""):
        self.fold_path = fold_path
        self.writer = SummaryWriter(hp['saved_model_path']+self.fold_path) # Start Tensorboard to track metrics
        self.training_cm_pdf = PdfPages(hp['saved_model_path'] + self.fold_path + 'Training_Confusion_Matrices.pdf')
        self.testing_cm_pdf = PdfPages(hp['saved_model_path'] + self.fold_path + 'Validation_Confusion_Matrices.pdf')

        self.performance_datapoints = ['training_loss_datapoints', 'testing_loss_datapoints', \
                    'training_accuracy_datapoints', 'testing_accuracy_datapoints', \
                        'training_f1_score_datapoints', 'testing_f1_score_datapoints']
        
        self.performance = pd.DataFrame(columns=self.performance_datapoints)

        self.best_accuracy = 0.0

    def create_cm(self, labels, predictions, class_names, epoch, epoch_acc, epoch_loss):
        # Create the Confusion matrix
        self.cm = confusion_matrix(y_true=labels.cpu(), y_pred=predictions.cpu()) 
        self.df_cm = pd.DataFrame(self.cm, index = [i for i in class_names], columns = [i for i in class_names])
        plt.figure(figsize = (10,10))
        plt.title(f'Confusion matrix for epoch {epoch}, Accuracy: {"{:.4f}".format(epoch_acc)}, Loss: {"{:.4f}".format(epoch_loss)}',\
                         loc='center')
        cm_img = sn.heatmap(self.df_cm, annot=True, fmt='g')
        self.fig = cm_img.get_figure()
    
    def get_metrics(self,labels, predictions, class_names, epoch, epoch_acc, epoch_loss,phase,preds):
        
        # Create the confusion matrix
        self.create_cm(labels, predictions, class_names, epoch, epoch_acc, epoch_loss)

        if phase == 'train':

            self.training_loss_datapoint = epoch_loss
            self.training_accuracy_datapoint = epoch_acc
            self.training_f1_score_datapoint = f1_score(labels.cpu().data, preds.cpu(), average='macro')
            self.training_cm_pdf.savefig(self.fig)

            # Report to Tensorboard Loss, Accuracy, F1-Score and Confusion Matrix of Training epoch
            self.writer.add_scalar('Training Loss', epoch_loss, epoch)
            self.writer.add_scalar('Training Accuracy', epoch_acc, epoch)
            self.writer.add_scalar('Training F1-Score', f1_score(labels.cpu().data, preds.cpu(), average='macro'), epoch)
            self.writer.add_figure("Training Confusion matrix", self.fig, epoch)
        else:
            
            # Record Test Statistics of Epoch
            self.testing_loss_datapoint = epoch_loss
            self.testing_accuracy_datapoint = epoch_acc
            self.testing_f1_score_datapoint = f1_score(labels.cpu().data, preds.cpu(), average='macro')
            self.testing_cm_pdf.savefig(self.fig)

            # Report to Tensorboard Loss, Accuracy, F1-Score and Confusion Matrix of Testing epoch
            self.writer.add_scalar('Testing Loss', epoch_loss, epoch)
            self.writer.add_scalar('Testing Accuracy', epoch_acc, epoch)
            self.writer.add_scalar('Testing F1-Score', f1_score(labels.cpu().data, preds.cpu(), average='macro'), epoch)
            self.writer.add_figure("Testing Confusion matrix", self.fig, epoch)

            if epoch_acc > self.best_accuracy:

                self.best_epoch = epoch
                self.best_accuracy = epoch_acc
                self.best_cm = self.fig

    def collect_epoch_performance(self, epoch):
        self.performance.loc[epoch] = [round(self.training_loss_datapoint,3), round(self.testing_loss_datapoint,3),\
                    round(self.training_accuracy_datapoint.cpu().item(),3), round(self.testing_accuracy_datapoint.cpu().item(),3),\
                        round(self.training_f1_score_datapoint,3), round(self.testing_f1_score_datapoint,3)]    

    def gather_all_metrics(self,hp, dataset_info, model_architecture):
        self.tags = ['Loss', 'Accuracy', 'F1-Score']
        create_plots(self.performance, self.tags, hp['saved_model_path']+self.fold_path)

        self.perfomance_recordings = {   
                            'perfomance_data' : self.performance,
                            'best_epoch': self.best_epoch,
                            'best_accuracy': self.best_accuracy,
                            'cm_image' : hp['saved_model_path'] + self.fold_path + "best_cm.jpg",
                            'loss_image': hp['saved_model_path'] + self.fold_path + self.tags[0] + ".jpg",
                            'acc_image': hp['saved_model_path'] + self.fold_path + self.tags[1] + ".jpg",
                            'f1_image' : hp['saved_model_path'] + self.fold_path + self.tags[2] + ".jpg"
                        }
        
        self.training_cm_pdf.close()
        self.testing_cm_pdf.close()
        self.fig.savefig(hp['saved_model_path'] + self.fold_path + "best_cm.jpg")
        create_report(hp, self.perfomance_recordings, dataset_info, model_architecture, fold_path=self.fold_path)


class PDF(FPDF):
    
    # Create header to pdf file
    def header(self):
        self.set_font('Arial', 'B', 30)
        self.cell(w=190, h = 10, txt = 'CNN REPORT', border = 1, ln = 2, align = 'C', fill = False, link = '')
        self.ln(10)

    # Create footer to pdf file
    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        # Select Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Print current and total page numbers
        self.cell(0, 10, 'Page %s' % self.page_no() + '/{nb}', 0, 0, 'C')
    
    # Create a chapter file
    def chapter_title(self, num, label):
        # Arial 12
        self.set_font('Arial', '', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, 'Chapter %d : %s' % (num, label), 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    # Create a chapter body
    def chapter_body(self, name):
        # Read text file
        try:
            with open(name, 'rb') as fh:
                txt = fh.read().decode('latin-1')
        except:
            txt=name

        # Times 12
        self.set_font('Times', '', 12)
        # Output justified text
        self.multi_cell(0, 5, txt)
        # Line break
        self.ln()

    # Print a chapter
    def print_chapter(self, num, title, name):
        self.add_page()
        self.chapter_title(num, title)
        self.chapter_body(name)
    
    # Create a plot
    def plot_chart(df, tag, saved_model_path):  
        df.plot(x ='Epochs', y=tag, kind = 'line')
        plt.savefig( saved_model_path + tag + '.jpg')
        plt.clf()
        plt.close()

def create_report(hp, perfomance_recordings, dataset_info, model, fold_path = ""):

    print("Creating Report...")
    if fold_path != '':
        fold = r'_' + fold_path.split('/')[0].split('_')[-2] + r'_' + fold_path.split('/')[0].split('_')[-1]
    else:
        fold = fold_path

    report_name = hp['saved_model_path'] + fold_path + 'model_report_' + hp['unique_id'] + fold + '.pdf'
    HEIGHT = 297
    WIDTH = 210
    pdf = PDF()

    data = perfomance_recordings['perfomance_data']
    best_epoch = perfomance_recordings['best_epoch']

    # Data info preparation

    chapter_1_model_info = ""
    chapter_2_performance_info = ""

    for key, value in hp.items():
        name = key.replace("_"," ")
        chapter_1_model_info = chapter_1_model_info + name[0].upper() + name[1:] + ': ' + str(value) + "\n"
    
    for key, value in dataset_info.items():
        name = key.replace("_"," ")
        chapter_1_model_info = chapter_1_model_info + name[0].upper() + name[1:] + ': ' + str(value) + "\n"

    chapter_2_performance_info = f"Best model perfomance occured in epoch {best_epoch}, where the following statitics were held:\n\n Training accuracy: {data['training_accuracy_datapoints'].loc[best_epoch]}\n\n Training loss: {data['training_loss_datapoints'].loc[best_epoch]}\n\n Training F1 Score{data['training_f1_score_datapoints'].loc[best_epoch]}\n\n Validation accuracy: {data['testing_accuracy_datapoints'].loc[best_epoch]}\n\n Validation loss: {data['testing_loss_datapoints'].loc[best_epoch]}\n\n Validation F1 Score: {data['testing_f1_score_datapoints'].loc[best_epoch]}\n\n"
    
    # CHAPTER 1
    pdf.set_title("custom neural model report")
    pdf.print_chapter(1, 'Model Summary', chapter_1_model_info)
    pdf.ln(10)

    #CHAPTER 2
    pdf.print_chapter(2, 'Model Architecture', model)
    pdf.ln(10)

    # CHAPTER 3
    pdf.print_chapter(3, 'Confusion Matrix', chapter_2_performance_info)
    pdf.ln(10)
    pdf.image(perfomance_recordings['cm_image'], x = 25, y = 105, w = WIDTH/1.2, h = 0, type = 'JPG', link = '')

    #CHAPTER 4
    pdf.print_chapter(4, 'Model Metrics', '')
    pdf.ln(10)
    
    pdf.image(perfomance_recordings['acc_image'], x = 0, y = 40, w = WIDTH, h = HEIGHT/2.8, type = 'JPG', link = '')
    pdf.ln(10)
    
    pdf.image(perfomance_recordings['loss_image'], x = 0, y = 150, w = WIDTH, h = HEIGHT/2.8, type = 'JPG', link = '')

    pdf.add_page()
    pdf.image(perfomance_recordings['f1_image'], x = 0, y = 40, w = WIDTH, h = HEIGHT/2.8, type = 'JPG', link = '')

    pdf.footer()
    pdf.alias_nb_pages()
    pdf.output(report_name)

    print(f"Report Ready! Saved at: {report_name}")
    print(hp['model_name'])

# Function to save the trained model
#
# Inputs: 
#          - model: the model to be trained
#          - path: path of the saved file
#
# Product: - Saves the model
#
# 
# Outputs: - None

def create_plots(dataframe, tags, path):
    
    columns = iter(dataframe)

    for index, column in enumerate(columns):
        plot = dataframe.plot(y=[column,next(columns)],
                            title = tags[index] + " perfomance",
                            grid=True,
                            xlabel = 'Epochs',
                            ylabel = tags[index],
                            kind = 'line',                
                            figsize=(20, 10))

        plot.title.set_size(18)
        fig = plot.get_figure()
        fig.savefig(path + tags[index] + '.jpg')