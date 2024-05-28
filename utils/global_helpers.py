import matplotlib.pyplot as plt
import yaml
import pandas as pd
import os
import re
with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

def show_results(name=None, path=None, history=None):
    #TODO Fix this method
    if not (path):
        path = f"results\{name}_train_history"

    #with open(path, 'rb') as file_pi:
        #history = pickle.load(file_pi)    
    print(history)
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    precision= history['precision']
    recall=history['recall']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs_range = range(model_params['epochs'])

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.plot(epochs_range, precision, label='Precision')
    plt.plot(epochs_range, recall, label="Recall")
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(f"results\{name}_history_graph.png")
    plt.show()

def save_history(history, name):
    """
    Saves the history of a model under results folder.
    Params from params.yaml

    Input:
    history - history object for model
    name - name of model
    """
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


    
#show_results(path="E:/results_ResNet50_train_history")