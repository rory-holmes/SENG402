import matplotlib.pyplot as plt
import numpy as np
from keras import layers, applications, metrics
from keras.optimizers import Adam
import sys
sys.path.append('utils')
import utils.video_process as vp
from keras.models import Sequential, Model, load_model
import yaml
import logging
from statistics import mean 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


with open("/params/model_params.yaml", "r") as f:
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
        history = self.model.fit(
            training_data,
            validation_data=validation_data,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
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
        
        print("Accuracy:", accuracy)
        print('Precision:', precision)
        print('Recall:', recall)


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
        self.name = "VGG_16"

        # Create the complete model
        self.model = Model(inputs=base_model.input, outputs=predictions)
