from keras.applications import InceptionResNetV2
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.models import Model
from keras import layers
import numpy as np

class PhasePredictor3DCNN:
    def __init__(self, input_shape, num_frames, num_classes=4):
        """
        Initializes the PhasePredictor3DCNN model.

        :param input_shape: Tuple (height, width, channels), the shape of the individual frames.
        :param num_frames: Number of frames to stack for 3D CNN input.
        :param num_classes: Number of output classes for the phase prediction.
        """
        self.input_shape = input_shape
        self.num_frames = num_frames
        self.num_classes = num_classes

        # Build the InceptionResNetV2 model without the last two dense layers and global average pooling layer
        self.feature_extractor = self.build_inception_feature_extractor()

        # Build the 3D-CNN model for phase prediction
        self.model = self.build_model()

    def build_inception_feature_extractor(self):
        """
        Builds the InceptionResNetV2 feature extractor without the last two dense layers and average pooling layer.

        :return: InceptionResNetV2 model as a feature extractor.
        """
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
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
        input_shape_3d = (self.num_frames, 6, 6, 1536)
        
        # Input layer
        inputs = Input(shape=input_shape_3d)
        
        # 3D convolutional layers
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(inputs)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        # Flatten and fully connected layers
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)  # 4 output classes for phases
        
        # Build the full model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def extract_features(self, frames):
        """
        Uses the feature extractor to extract features from the input frames.

        :param frames: Input frames (batch of images) to extract features from.
        :return: Extracted features from the InceptionResNetV2 model.
        """
        return self.feature_extractor.predict(frames)

    def stack_features(self, features, num_frames):
        """
        Stacks features into 3D blocks (sequence of frames) for input to 3D CNN.

        :param features: Features extracted from individual frames.
        :param num_frames: Number of frames to stack.
        :return: Stacked 3D features.
        """
        stacked_features = []
        
        # Stack consecutive features into 3D blocks
        for i in range(len(features) - num_frames + 1):
            stacked_features.append(features[i:i + num_frames])
        
        return np.array(stacked_features)

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=20):
        """
        Extracts features, stacks them, and trains the 3D-CNN model.

        :param X_train: Training frames (batch of images).
        :param y_train: Training labels (one-hot encoded phases).
        :param X_val: Validation frames (batch of images).
        :param y_val: Validation labels (one-hot encoded phases).
        :param batch_size: Batch size for training.
        :param epochs: Number of epochs for training.
        """
        # Step 1: Extract features for each frame
        print("Extracting features for training data...")
        train_features = np.array([self.extract_features(frames) for frames in X_train])
        val_features = np.array([self.extract_features(frames) for frames in X_val])

        # Step 2: Stack features into 3D blocks for 3D-CNN input
        print("Stacking features for 3D CNN input...")
        X_train_stacked = self.stack_features(train_features, self.num_frames)
        X_val_stacked = self.stack_features(val_features, self.num_frames)

        # Step 3: Train the model
        print("Training 3D CNN model...")
        self.model.fit(X_train_stacked, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val_stacked, y_val))


    def predict(self, X_test):
        """
        Predicts the phase for the given test data.

        :param X_test: Test data (3D-CNN inputs).
        :return: Predicted phases.
        """
        return self.model.predict(X_test)
