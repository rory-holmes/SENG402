import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import layers
from keras.models import Sequential
import yaml
import logging

with open("params\model_params.yaml", "r") as f:
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

    def train(self):
        logging.info(f" {self.name}: Training started")
        self.model.compile(
            optimizer='adam',
            loss="binary_crossentropy",
            metrics=['accuracy']
            )
        history = self.model.fit(
            x=self.training_data,
            validation_data=self.validation_data,
            epochs=self.epochs
        )
        logging.info(f" {self.name}: Training completed")
        return self.model.evaluate(self.validation_data)
        #TODO fix: ValueError: `labels.shape` must equal `logits.shape` except for the last dimension. Received: labels.shape=(7,) and logits.shape=(1, 7)

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
