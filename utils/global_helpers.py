import matplotlib.pyplot as plt
import yaml
import pandas as pd
import os
import utils.video_process as vp
import re
import numpy as np
import cv2
from keras.models import load_model
from matplotlib.ticker import MaxNLocator

with open("params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

def graph_results(csv_path1, csv_path2, csv_path3):
    """
    Plots the validation accuracy from three csv files on a single graph with different colors.
    
    Inputs:
    csv_path1, csv_path2, csv_path3 - CSV files containing results to be plotted
    """
    # Load the CSV files
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)
    df3 = pd.read_csv(csv_path3)
    
    # Plot the data
    epochs1 = range(1, len(df1) + 1)
    epochs2 = range(1, len(df2) + 1)
    epochs3 = range(1, len(df3) + 1)

    plt.figure(figsize=(14, 8))
    
    # Validation Accuracy plot
    plt.plot(epochs1, df1['val_loss'], 'b', label='InceptionResnetV2')
    plt.plot(epochs2, df2['val_loss'], 'r', label='ResNet50')
    plt.plot(epochs3, df3['val_loss'], 'g', label='VGG16')
    plt.title('Validation Loss over 3 epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
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
    