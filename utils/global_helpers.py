import matplotlib.pyplot as plt
import yaml
import pandas as pd
import os
import sys
import video_process as vp
import re
import numpy as np
import cv2
from keras.models import load_model

with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

def show_results(csv_path):
    """
    Plots train history [accuracy, F1Score, precision, recall] from a csv file given
    
    Inputs:
    csv_path - csv file containg results to be plotted
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Calculate F1 Score for validation data
    df['val_f1_score'] = 2 * (df['val_precision'] * df['val_recall']) / (df['val_precision'] + df['val_recall'])
    
    # Plot the data
    epochs = range(1, len(df) + 1)

    plt.figure(figsize=(14, 10))
    
    # Loss plot
    plt.subplot(3, 1, 1)
    plt.plot(epochs, df['loss'], 'b', label='Training loss')
    plt.plot(epochs, df['val_loss'], 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(3, 1, 2)
    plt.plot(epochs, df['accuracy'], 'b', label='Training accuracy')
    plt.plot(epochs, df['val_accuracy'], 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # F1 Score plot
    plt.subplot(3, 1, 3)
    plt.plot(epochs, df['val_f1_score'], 'g', label='Validation F1 Score')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def save_history(history, name):
    """
    Saves the history of a model under results folder.
    Params from params.yaml

    Inputs:
    history - history object for model
    name - name of model
    """
    print(history)
    pattern = r'\((\d+)\)'
    history_df = pd.DataFrame(history.history)
    history_num = 0
    history_name = f"{name}({history_num})_train-history"
    for file in os.listdir(params['results_path']):
        if file == history_name:
            num = int(re.findall(pattern(file))[0])
            if num > history_num:
                history_num = num + 1
    
    history_name = f"{name}({history_num})_train-history"
    history_df.to_csv(os.path.join(params['results_path'], history_name), index=False)

def demo(model_path):
    """
    Retrieves frames, labels from generator, evaluates with model, displays image and evaluations.

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
    
#show_results(path="E:/results_ResNet50_train_history")
demo("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/results/ResNet50(0).keras")