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
import os
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
    def __init__(self, base_model, name):
        self.name = gh.get_logger_name(name)
        self.training_data = []
        self.validation_data = []
        self.model = None
        self.epochs = model_params['epochs']
        self.batch_size = model_params['batch_size']
        self.num_classes = model_params["num_classes"]
        self.learning_rate = model_params['learning_rate']

        freeze_layers(base_model)

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
        csv_logger = CSVLogger(os.path.join(params['results_path'], f"{self.name}_training-history.log"))
        history = self.model.fit(
            training_data,
            validation_data=validation_data,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[UnfreezeOnMinLoss(), csv_logger]
        )
        self.model.save(os.path.join(params['results_path'], f"{self.name}.keras"))

    def test(self, made_model=None):
        # If using a premade model, not integrated into pipeline
        if made_model != None:
            self.model = load_model(made_model)
        csv_logger = CSVLogger(f"{self.name}_testing-history.log")
        self.model.evaluate(vp.data_generator("testing", self.batch_size), callbacks=[csv_logger])

class ResNet50_Model(CNN):
    def __init__(self, pretrained_weights="imagenet"):
        base_model = applications.ResNet50(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=None,
            input_shape=(img_height, img_width, channels),
            pooling=None,
            classes=7,
            classifier_activation="softmax",
        )
        super().__init__(base_model, "ResNet50")

class InceptionResNetV2_Model(CNN):
    def __init__(self, pretrained_weights="imagenet"):
        base_model = applications.InceptionResNetV2(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=None,
            input_shape=(img_height, img_width, channels),
            pooling=None,
            classes=7,
            classifier_activation="softmax",
        )
        super().__init__(base_model, "InceptionResNetV2")

class VGG16_Model(CNN):
    def __init__(self, pretrained_weights="imagenet"):
        base_model = applications.VGG16(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=None,
            input_shape=(img_height, img_width, channels),
            pooling=None,
            classes=7,
            classifier_activation="softmax",
        )
        super().__init__(base_model, "VGG_16")

class UnfreezeOnMinLoss(callbacks.Callback):
    """
    Unfreeze's models layers when loss is at its min, if performance decreases after freezing, stops training.

    Inputs:
    patience - Number of epochs to wait after min has been hit. After this number of no improvment, unfreezes layers or halts training.
    unfreeze_layers - Amount of layers to unfreeze after patience has been used.
    """
    def __init__(self):
        self.patience = model_params['patience']
        self.best_weights = None
        self.current_frozen = 0
        self.unfreeze_more = False

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs['layers_unfrozen'] = self.current_frozen
        logs['restored_weights_to'] = False
        current = logs.get("val_loss")
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
                    self.unfreeze_layers()
                    self.unfreeze_more = False
                else:
                    self.model.stop_training = True
                    logging.info(f"Restoring model weights from the end of the best epoch, stopped at epoch {self.stopped_epoch}")
                    logs['restored_weights_to'] = self.stopped_epoch
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

def freeze_layers(network):
    # Call before compiling
    for layer in network.layers:
        layer.trainable = False