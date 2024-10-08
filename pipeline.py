import yaml 
import sys
sys.path.append('utils')
sys.path.append('models')
import utils.file_helpers as fh
import utils.global_helpers as gh
from models.CNN import *
import tensorflow as tf
import logging

with open("params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

def testing(model_path):
    """
    Tests premade model, saves under results folder

    Inputs:a
    model_path - path to premade model

    """
    #load_model(model_path).evaluate(vp.data_generator("testing", model_params['batch_size']))
    test_with_predict(model_path)

def forward_pass(model):
    """
    Forward pass of model training

    Inputs:
    model - model to be trained
    """
    fh.split_data()
    model.train()
    logging.info(f"Testing {str(model.name)}")
    model.test()

def train_models(models):
    """
    Calls forward pass for each model, clears session between model training
    
    Inputs:
    models - List of models to be trained.
    """
    for m in models:
        tf.keras.backend.clear_session()
        forward_pass(m())

#===========================================================================

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    train_models([InceptionResNetV2_Model])
    #testing("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/results/InceptionResNetV2(1).keras")