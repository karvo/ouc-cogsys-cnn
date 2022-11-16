import numpy as np
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Class for the creation of the reports and the statistics
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

def report(hp, perfomance_recordings):

    HEIGHT = 297
    WIDTH = 210
    pdf = PDF()

    for i in hp:
        name = list(hp)[i].split('_').title()
        value = str(list(hp.values())[i])
        model_summary = model_summary + name + ": " + value

    pdf.set_title("custom neural model report")
    pdf.print_chapter(1, 'Model Summary', model_summary)

    pdf.ln(10)

    cm_results = f"Best perfomance occured in epoch {perfomance_recordings['best_epoch']},\
                    where the following statitics were held:\n\n \
                        Training accuracy: {perfomance_recordings['best_training_accuracy']:.2f}\n\n \
                            Training loss: {perfomance_recordings['best_training_loss']:.2f}\n\n \
                                Training F1 Score{perfomance_recordings['best_training_f1']:.2f}\n\n \
                                    Validation accuracy: {perfomance_recordings['best_accuracy']:.2f}\n\n \
                                        Validation loss: {perfomance_recordings['best_loss']:.2f}\n\n \
                                            Validation F1 Score{perfomance_recordings['best_testing_f1']:.2f}\n\n"

    pdf.print_chapter(2, 'Confusion Matrix', cm_results)
    pdf.ln(10)
    pdf.image(perfomance_recordings['best_cm'], x = 25, y = 130, w = WIDTH/1.4, h = 0, type = 'JPG', link = '')

    pdf.print_chapter(3, 'Model Metrics', '')
    pdf.ln(10)
    
    imgs = []

    for tag in perfomance_recordings['perfomance_data'].columns:
        img = pdf.plot_chart(perfomance_recordings['perfomance_data'], tag, hp['saved_model_path'])
        imgs.append(img)
    
        pdf.image(hp['saved_model_path'] + "Accuracy.jpg", x = 0, y = 35, w = WIDTH, h = 125, type = 'JPG', link = '')   

        pdf.image(hp['saved_model_path'] + "Loss.jpg", x = 0, y = 160, w = WIDTH, h = 125, type = 'JPG', link = '')  
        pdf.add_page() #add a page
        pdf.image(hp['saved_model_path'] + "F1-Score.jpg", x = 0, y = 35, w = WIDTH, h = 125, type = 'JPG', link = '')   

        pdf.footer()
        pdf.alias_nb_pages()
        pdf.output(hp['saved_model_path'] + 'model_report_' + hp['unique_id'] + '.pdf', 'F')
