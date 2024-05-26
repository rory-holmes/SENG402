import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import layers, applications, metrics
from keras.optimizers import Adam
from keras.models import Sequential, Model, load_model
import yaml
import logging
import os
from statistics import mean 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


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
            metrics=['accuracy', metrics.Precision(), metrics.Recall()
            ]
        )

    def train(self, dataset_func):
        """
        Trains model on dataset
        """
        self.compile()
        logging.info("Extracting training data")
        training_data = dataset_func('training', self.batch_size)
        logging.info("Extracting validation data")
        validation_data = dataset_func('validation', self.batch_size)
        logging.info("Training model")
        history = self.model.fit(
            training_data,
            validation_data=validation_data,
            epochs=self.epochs
        )
        self.model.save(f"{self.name}.keras")
        return history, self.name

    def test(self, dataset_func, made_model=None):
        # If using a premade model, not integrated into pipeline
        if made_model == None:
            model = self.model
        else:
            model = load_model(made_model)
        true_labels = []
        predictions = []
        m_predictions = []
        count = 0 #TODO Remove 
        # Loop through testing data
        for X_batch, y_batch in dataset_func("testing", self.batch_size):
            y_pred_prob = model.predict(X_batch)
            y_pred = (y_pred_prob > 0.5).astype(int)
            m_y_pred = np.argmax(y_pred_prob, axis=1)  # For multi-class classification
            true_labels.extend(y_batch)
            predictions.extend(y_pred)
            m_predictions.extend(m_y_pred)
            if count >= 1000:
                break
            count += 1

        print("TL:",true_labels)
        print("Pr:",predictions)
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)

        #conf_matrix = confusion_matrix(true_labels, m_predictions)
        #precision = precision_score(true_labels, predictions, average='binary')  # For binary classification
        #recall = recall_score(true_labels, predictions, average='binary')
        accuracy = accuracy_score(true_labels, predictions)
        # For multi-class classification:
        m_precision = precision_score(true_labels, predictions, average='macro')
        m_recall = recall_score(true_labels, predictions, average='macro')

        #print('Confusion Matrix:\n', conf_matrix)
        #print('Precision:', precision)
        #print('Recall:', recall)
        print("Accuracy:", accuracy)
        print("Multi:")
        print('Precision:', m_precision)
        print('Recall:', m_recall)

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

class InceptionResNetV2_Model(Base_Model):
    def __init__(self, pretrained_weights="imagenet"):
        super().__init__()
        base_model = applications.InceptionResNetV2(
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
        self.name = "InceptionResNetV2"

class VGG16_Model(Base_Model):
    def __init__(self, pretrained_weights="imagenet"):
        super().__init__()
        base_model = applications.VGG16(
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
        self.name = "VGG16"