import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import layers, applications
from keras.optimizers import Adam
from keras.models import Sequential, Model
import yaml
import logging
import os
from statistics import mean 

with open("params\model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

class Base_Model:
    def __init__(self):
        self.training_data = []
        self.validation_data = []
        self.model = None
        self.epochs = model_params['epochs']
        self.batch_size = model_params['batch_size']
        self.img_height = model_params["image_height"]
        self.img_width = model_params["image_width"]
        self.channels = model_params["channels"]
        self.num_classes = model_params["num_classes"]
        self.learning_rate = model_params['learning_rate']
        self.name = "NULL"

    def compile(self):
        """
        Compiles model.
        Params from model_params.yaml
        """
        
        self.model.compile(
            optimizer= Adam(learning_rate=self.learning_rate),
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
            training_data,
            validation_data=validation_data,
            epochs=self.epochs
        )
        self.model.save(f"{self.name}.keras")
        return history
    
    def train_on_batch(self, dataset):
        self.compile()
        history = []
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            self.model.reset_metrics()
            batch_loss = []
            batch_accuracy = []
            for batch_frames, batch_annotations in dataset("training", self.batch_size):
                # Train on batch
                loss, accuracy = self.model.train_on_batch(batch_frames, batch_annotations)
                batch_loss.append(loss)
                batch_accuracy.append(accuracy)
            # Print batch metrics (optional)
            print(f"Epoch Loss: {mean(batch_loss):.4f}, Epoch Accuracy: {mean(batch_accuracy):.4f}")
            # Evaluate on validation data after each epoch
            val_loss, val_accuracy = self.evaluate_on_batch(dataset)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            history.append(f"Epoch {epoch+1}:   Loss: {mean(batch_loss)}, Accuracy: {mean(batch_accuracy)}, val_loss: {val_loss}, val_accuracy: {val_accuracy}")
       
        self.model.save(f"{self.name}.keras")
        return history
    def evaluate_on_batch(self, dataset):
            # Initialize metrics
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        validation_steps = len(os.listdir("validation\annotations"))
        val_generator = dataset("validation", self.batch_size)
        # Iterate over validation batches
        for _ in range(validation_steps):
            val_batch_frames, val_batch_annotations = next(val_generator)
            
            # Evaluate on batch
            loss, accuracy = self.model.evaluate(val_batch_frames, val_batch_annotations, batch_size=self.batch_size)
            
            # Accumulate metrics
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1
        
        # Calculate average metrics
        average_loss = total_loss / num_batches
        average_accuracy = total_accuracy / num_batches
        
        return average_loss, average_accuracy
    
class Sequential_Model(Base_Model):
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

class ResNet50_Model(Base_Model):
    def __init__(self, pretrained_weights="imagenet"):
        super().__init__()
        base_model = applications.ResNet50(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=None,
            input_shape=(self.img_height, self.img_width, self.channels),
            pooling=None,
            classes=7,
            classifier_activation="softmax",
        )
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)  # Add a fully connected layer
        predictions = layers.Dense(7, activation='softmax')(x)  # Add the final output layer for 7 classes

        # Create the complete model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.name = "ResNet50"