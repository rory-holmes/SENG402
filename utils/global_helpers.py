import matplotlib.pyplot as plt
import yaml
import pickle
with open("params\model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

def show_results(name):
    with open(f"\results\{name}_train_history", 'wb') as file_pi:
        history = pickle.load(file_pi)    

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    binary_cross_entropy= history.history['BinaryCrossentropy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(model_params['epochs'])

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.plot(epochs_range, binary_cross_entropy, label='BinaryCrossentropy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(f"\results\{name}_history_graph.png")
    plt.show()

def save_history(history,  name):
    path = f"\results\{name}_train_history"
    with open(path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
