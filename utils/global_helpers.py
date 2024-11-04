import matplotlib.pyplot as plt
import yaml
import pandas as pd
import os
import re
import numpy as np
import cv2
from keras.models import load_model
from matplotlib.ticker import MaxNLocator
import ast
with open("params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)
tools=['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']
def graph_results(csv_path):
    """
    Plots the validation accuracy from three csv files on a single graph with different colors.
    
    Inputs:
    csv_path1, csv_path2, csv_path3 - CSV files containing results to be plotted
    """
    data = {
        'epoch': [0, 1, 2, 3],
        'val_accuracy': [0.23, 0.256, 0.253, 0.295],
        'val_loss': [3.2, 3.04, 2.53, 1.45],
        'val_precision': [0.98, 0.985, 0.976, 0.98],
        'val_recall': [0.0124, 0.0167, 0.0234, 0.242]
    }

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Plot val_accuracy across epochs
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['val_accuracy'], marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Across Epochs')
    plt.ylim(0, 1)
    plt.grid(True)

    # Display the plot
    plt.show()

def get_logger_name(name):
    """
    Iterates over saved models to return a unique file name for logging.
    Params from params.yaml

    Inputs:
    name - name of model
    """
    pattern = r'\((\d+)\)'
    history_num = 0
    for file in os.listdir(params['results_path']):
        if name in file:
            num = int(re.findall(pattern, file)[0])
            if num >= history_num:
                history_num = num + 1
    
    history_name = f"{name}({history_num})"
    return history_name

def demo(model_path):
    """
    Retrieves frames, labels from generator, evaluates with model, displays image and evaluations.
    Frames iterated by 'n' key. Frame saved by 'enter' key. Quit by 'q' key.

    Inputs:
    model_path - Path to model to be demoed
    """
    model = load_model(model_path)
    org = (50, 50)  # Bottom-left corner of the text string in the image
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 0.4  # Font scale factor
    color = (255, 0, 0)  # Color in BGR (Blue, Green, Red)
    thickness = 2  # Thickness of the lines used to draw a text
    for X_batch, y_batch in vp.data_generator("testing", 1):
        y_pred_prob = model.predict(X_batch)
        y_pred = np.where(y_pred_prob > 0.5, 1, 0)
        
        cv2.putText(X_batch[0], f"Predicted: {y_pred}", org, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(X_batch[0], f"Actual: {y_batch}", (50, 100), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Frame', X_batch[0])        

            # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()
graph_results('Boom')
#graph_results(r"C:\Users\Rory\OneDrive - University of Canterbury\Desktop\University\Year 4\SENG402\results\inceptionResults.log")