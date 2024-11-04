import yaml 
import sys
sys.path.append('utils')
sys.path.append('models')
import utils.file_helpers as fh
import utils.global_helpers as gh
from models.CNN import *
from models.PhaseCNN import *
import tensorflow as tf
import logging

with open("params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

def train_feature_extractor(model):
    """
    Forward pass of model training

    Inputs:
    model - model to be trained

    Returns:
    model - The trained model
    """
    fh.split_data()
    model.train()
    logging.info(f"Testing {str(model.name)}")
    model.test()
    return model

def train_phase_detector(base_model):
    """
    Trains the phase detection model with the given feature extractor:
    
    Inputs:
    base_model - Trained Feature extractor

    Returns:
    model - The trained model
    """
    fh.split_phase_data()
    model = PhasePredictor3DCNN(base_model)
    model.train()
    return model

def full_training_cycle():
    """
    Trains both the feature extractor and phase detector
    """
    feature_extractor = train_feature_extractor(InceptionResNetV2_Model)
    phase_detector = train_phase_detector(feature_extractor)
    return phase_detector

#===========================================================================

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    full_training_cycle()