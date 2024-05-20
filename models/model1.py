import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import layers
from keras.models import Sequential
import yaml
import logging

with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

class Model:
    def __init__(self):
        self.training_data = []
        self.validation_data = []
        self.model = None
        self.epochs = model_params['epochs']
        self.img_height = model_params["image_height"]
        self.img_width = model_params["image_width"]
        self.channels = model_params["channels"]
        self.num_classes = model_params["num_classes"]
        self.name = "NULL"

    def compile(self):
        """
        Compiles model.
        Params from model_params.yaml
        """
        self.model.compile(
            optimizer= model_params['optimizer'],
            loss= model_params["loss"],
            metrics= model_params['metrics']
        )

    def train(self, dataset_func):
        """
        Trains model on dataset
        """
        self.compile()
        logging.info("Extracting training data")
        training_data = dataset_func('training')
        logging.info("Extracting validation data")
        validation_data = dataset_func('validation')
        logging.info("Training model")
        history = self.model.fit(
            x=training_data,
            validation_data=validation_data,
            epochs=self.epochs
        )
        return history

class Sequential_Model(Model):
    def __init__(self):
        super().__init__()
        self.model = Sequential([
            layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, self.channels)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes)
        ])
        self.name = "Sequential"
