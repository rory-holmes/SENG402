import numpy as np
from keras import layers, applications, metrics, callbacks
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
import sys
sys.path.append('utils')
sys.path.append('params')
import utils.video_process as vp
import utils.global_helpers as gh
from keras.models import Model, load_model
import yaml
import logging
from sklearn.metrics import precision_score, recall_score, accuracy_score

with open("params\model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

img_height = model_params["image_height"]
img_width = model_params["image_width"]
channels = model_params["channels"]
    
class CNN:
    """
    Class to be passed pre-trained models, initialises values based on the model_params file.
    Contains methods for compiling, training, and testing.
    """
    def __init__(self, base_model):
        self.training_data = []
        self.validation_data = []
        self.model = None
        self.epochs = model_params['epochs']
        self.batch_size = model_params['batch_size']
        self.num_classes = model_params["num_classes"]
        self.learning_rate = model_params['learning_rate']

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)  # Add a fully connected layer
        predictions = layers.Dense(self.num_classes, activation='softmax')(x)  # Add the final output layer for 7 classes

        # Create the complete model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
    def compile(self):
        """
        Compiles model.
        Params from model_params.yaml
        """
        
        self.model.compile(
            optimizer= Adam(learning_rate=self.learning_rate),
            loss= model_params["loss"],
            metrics=['accuracy', metrics.Precision(), metrics.Recall()
            ]
        )

    def train(self):
        """
        Trains model on dataset
        """
        self.compile()
        steps_per_epoch, validation_steps = vp.get_training_validation_steps()
        logging.info("Extracting training data")
        training_data = vp.data_generator('training', self.batch_size)
        logging.info("Extracting validation data")
        validation_data = vp.data_generator('validation', self.batch_size)
        logging.info(f"Training model: {self.name}")
        csv_logger = CSVLogger(f"{gh.get_logger_name(self.name)}.log")
        history = self.model.fit(
            training_data,
            validation_data=validation_data,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[UnfreezeOnMinLoss(), csv_logger]
        )
        self.model.save(f"{self.name}.keras")
        return history, self.name

    def test(self, made_model=None):
        # If using a premade model, not integrated into pipeline
        if made_model == None:
            model = self.model
        else:
            model = load_model(made_model)

        true_labels = []
        predictions = []
        # Loop through testing data
        for X_batch, y_batch in vp.data_generator("testing", self.batch_size):
            y_pred_prob = model.predict(X_batch)
            y_pred = np.where(y_pred_prob > 0.5, 1, 0)
            true_labels.extend(y_batch)
            predictions.extend(y_pred)

        true_labels = np.array(true_labels)
        predictions = np.array(predictions)

        accuracy = accuracy_score(true_labels, predictions)
        # For multi-class classification:
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        gh.save_history(np.array([accuracy, precision, recall]).T, labels=['Accruacy', 'Precision', 'Recall'])

class ResNet50_Model(CNN):
    def __init__(self, pretrained_weights="imagenet"):
        self.name = "ResNet50"
        base_model = applications.ResNet50(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=None,
            input_shape=(img_height, img_width, channels),
            pooling=None,
            classes=7,
            classifier_activation="softmax",
        )
        super().__init__(base_model)

class InceptionResNetV2_Model(CNN):
    def __init__(self, pretrained_weights="imagenet"):
        self.name = "InceptionResNetV2"
        base_model = applications.InceptionResNetV2(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=None,
            input_shape=(img_height, img_width, channels),
            pooling=None,
            classes=7,
            classifier_activation="softmax",
        )
        super().__init__(base_model)

class VGG16_Model(CNN):
    def __init__(self, pretrained_weights="imagenet"):
        self.name = "VGG_16"
        base_model = applications.VGG16(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=None,
            input_shape=(img_height, img_width, channels),
            pooling=None,
            classes=7,
            classifier_activation="softmax",
        )
        super().__init__(base_model)

class UnfreezeOnMinLoss(callbacks.Callback):
    """
    Unfreeze's models layers when loss is at its min, if performance decreases after freezing, stops training.

    Inputs:
    patience - Number of epochs to wait after min has been hit. After this number of no improvment, unfreezes layers or halts training.
    unfreeze_layers - Amount of layers to unfreeze after patience has been used.
    """
    def __init__(self, patience=0, unfreeze_layers=0):
        self.patience = patience
        self.best_weights = None
        self.current_frozen = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs['layers_unfrozen'] = self.current_frozen
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.unfreeze_more = True
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.unfreeze_more:
                    self.model = self.unfreeze_layers()
                    self.unfreeze_more = False
                else:
                    self.model.stop_training = True
                    logging.info(f"Restoring model weights from the end of the best epoch, stopped at epoch {self.stopped_epoch}")
                    self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        return super().on_train_end(logs)

    def unfreeze_layers(self):
        """
        Freezes 'n_to_unfreeze' more layers of the network for finetuning
        """
        self.current_frozen += int(model_params['n_to_unfreeze'])
        for layer in self.model.layers[-self.current_frozen:]:
            layer.trainable = True

def freeze_layers(network, n=300):

    # Call before compiling
    for layer in network.layers[:n]:
        layer.trainable = False