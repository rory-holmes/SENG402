import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import layers
from keras.models import Sequential
import yaml
import sys

with open("params\model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

img_height = model_params["image_height"]
img_width = model_params["image_width"]
channels = model_params["channels"]
num_classes = model_params["num_classes"]
def get_sequential_model():
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, channels)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])
    return model

class Model:
    def __init__(self, get_model=get_sequential_model):
        self.training_data = []
        self.validation_data = []
        self.epochs = model_params['epochs']
        self.model = get_model()

    def train(self):
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
            )
        history = self.model.fit(
            x=self.training_data,
            validation_data=self.validation_data,
            epochs=self.epochs
        )
        print(self.model.evaluate(self.validation_data))
        #TODO fix: ValueError: `labels.shape` must equal `logits.shape` except for the last dimension. Received: labels.shape=(7,) and logits.shape=(1, 7)

