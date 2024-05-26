import matplotlib.pyplot as plt
import yaml
import pickle
with open("params/model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

def show_results(name=None, path=None):
    if not (path):
        path = f"results\{name}_train_history"
    with open(path, 'rb') as file_pi:
        history = pickle.load(file_pi)    
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

def save_history(history,  name):
    path = f"results\{name}_train_history"
    with open(path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


    
show_results(path="E:/results_ResNet50_train_history")