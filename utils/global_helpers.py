import matplotlib.pyplot as plt
import yaml
import pandas as pd
import os
import re
import numpy as np
import cv2
from keras.models import load_model
import video_process as vp

with open(r"params\\paths.yaml", "r") as f:
    paths = yaml.load(f, Loader=yaml.SafeLoader)

TOOLS=['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']
def graph_results(csv_path):
    """
    Plots the training results found in the csv_path
    
    Inputs:
        csv_path: Path containing the training history of the model
    """

    df = pd.DataFrame(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['val_accuracy'], marker='o', linestyle='-', color='b')

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Across Epochs')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

def get_logger_name(name):
    """
    Iterates over saved models to return a unique file name for logging.
    Params from paths.yaml

    Inputs:
        name - name of model
    """
    pattern = r'\((\d+)\)'
    history_num = 0
    for file in os.listdir(paths['results_path']):
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
    org = (50, 50)  
    font = cv2.FONT_HERSHEY_SIMPLEX  
    font_scale = 0.4  
    color = (255, 0, 0)  
    thickness = 2  
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

