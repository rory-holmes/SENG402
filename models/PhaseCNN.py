from keras.applications import InceptionResNetV2
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, TimeDistributed, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model, load_model
from keras import layers
import numpy as np
import sys
sys.path.append('utils')
sys.path.append('params')
import utils.video_process as vp
import utils.global_helpers as gh
import os
import yaml
from keras.callbacks import CSVLogger

with open("params/model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

with open("params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)
class PhasePredictor3DCNN:
    def __init__(self, base_model, num_classes=4):
        """
        Initializes the PhasePredictor3DCNN model.

        :param input_shape: Tuple (height, width, channels), the shape of the individual frames.
        :param num_frames: Number of frames to stack for 3D CNN input.
        :param num_classes: Number of output classes for the phase prediction.
        """
        self.feature_extractor = self.build_feature_extractor(base_model)
        self.num_frames = model_params['stack_size']
        self.num_classes = num_classes


        # Build the 3D-CNN model for phase prediction
        self.model = self.build_model()

    def build_feature_extractor(self, base_model):
        """
        Builds the InceptionResNetV2 feature extractor without the last two dense layers and average pooling layer.

        :return: InceptionResNetV2 model as a feature extractor.
        """        
        base_model = load_model(base_model)
        # Remove the global average pooling and final dense layers
        feature_output = base_model.get_layer('conv_7b_ac').output  # Shape: (None, 6, 6, 1536)
        
        # Create a new model with the output of 'conv_7b_ac'
        feature_extractor = Model(inputs=base_model.input, outputs=feature_output)
        
        return feature_extractor

    def build_model(self):
        """
        Builds the 3D-CNN phase prediction model using the extracted features from InceptionResNetV2.

        :return: The complete model for phase prediction.
        """
        # Input shape for 3D-CNN (num_frames, feature_height, feature_width, feature_channels)
        input_shape_2d = (self.num_frames, 6, 6, 1536)  # 6x6 is the InceptionResNetV2 feature maps for each frame

        # Input layer
        inputs = Input(shape=input_shape_2d)

        # TimeDistributed Conv2D block 1 (applied per frame using TimeDistributed)
        x = TimeDistributed(Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"))(inputs)
        x = TimeDistributed(MaxPooling2D(pool_size=2, padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)

        # TimeDistributed Conv2D block 2
        x = TimeDistributed(Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=2, padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)

        # TimeDistributed Conv2D block 3
        x = TimeDistributed(Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=2, padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)

        # TimeDistributed Conv2D block 4
        x = TimeDistributed(Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=2, padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)

        # Global Average Pooling (applied per frame using TimeDistributed)
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        # Fully connected layer with Dropout (applied per frame using TimeDistributed)
        x = TimeDistributed(Dense(units=512, activation="relu"))(x)
        x = TimeDistributed(Dropout(0.3))(x)

        # Output layer (applied per frame using TimeDistributed)
        outputs = TimeDistributed(Dense(units=self.num_classes, activation="softmax"))(x)

        # Define the model
        model = Model(inputs, outputs, name="2dcnn_phase_predictor")

        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

        return model
    def extract_features(self, frame):
        """
        Uses the feature extractor to extract features from the input frames.

        :param frames: Input frames (batch of images) to extract features from.
        :return: Extracted features from the InceptionResNetV2 model.
        """
        return self.feature_extractor.predict(frame)

    def stack_features(self, features, labels, num_frames):
        """
        Stacks features into 3D blocks (sequence of frames) for input to 3D CNN.

        :param features: Features extracted from individual frames.
        :param num_frames: Number of frames to stack.
        :return: Stacked 3D features.
        """
        stacked_features = []
        stacked_labels = []
        
        for i in range(0,len(features),num_frames):
            stacked_features.append(features[i:i+num_frames])
            stacked_labels.append(labels[i:i+num_frames])
        return np.array(stacked_features), np.array(stacked_labels)

    def train(self):
        """
        Trains the 3D-CNN model using a generator that yields batches of raw frames for training and validation.
        The method extracts features from the raw frames, stacks them, and trains the 3D-CNN model.

        :return: The trained 3D-CNN model.
        """
        csv_logger = CSVLogger(os.path.join(params['results_path'], f"3dcnn_training-history.log"))
        # Training and validation data generators
        train_generator = vp.phase_generator("training")
        val_generator = vp.phase_generator("validation")

        # Define steps per epoch based on the amount of training and validation data
        steps_per_epoch_train, steps_per_epoch_val = vp.get_phase_training_validation_steps()

        def process_batch(generator):
            """
            Processes a batch of raw frames from the generator:
            - Extracts features from the frames.
            - Stacks the features into 3D blocks for 3D-CNN input.
            
            :param generator: Generator yielding raw frames and corresponding labels.
            :return: Processed feature batch and corresponding labels.
            """
            while True:
                for raw_frames, labels in generator:
                    # Step 1: Extract features for each frame in the batch
                    features = np.array(self.extract_features(raw_frames))
                    # Step 2: Stack features into 3D blocks
                    stacked_features, stacked_labels = self.stack_features(features, labels, self.num_frames)
                    yield stacked_features, stacked_labels

        # Wrap the generators with the feature extraction and stacking logic
        train_data = process_batch(train_generator)
        val_data = process_batch(val_generator)

        # Train the model using the processed data
        self.model.fit(
            train_data,
            steps_per_epoch=steps_per_epoch_train,
            validation_data=val_data,
            validation_steps=steps_per_epoch_val,
            epochs=20,  # Number of epochs can be adjusted
            callbacks=[csv_logger],
            batch_size=8
        )

        # Return the trained model
        self.model.save(os.path.join(params['results_path'], "3dCNN.keras"))
        return self.model


    def predict(self, X_test):
        """
        Predicts the phase for the given test data.

        :param X_test: Test data (3D-CNN inputs).
        :return: Predicted phases.
        """
        return self.model.predict(X_test)
