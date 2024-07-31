import yaml 
import sys
sys.path.append('utils')
sys.path.append('models')
import utils.file_helpers as fh
import utils.video_process as vp
import utils.global_helpers as gh
from models.model1 import *
import tensorflow as tf
import os
import logging

with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

def forward_pass(model):
    fh.split_data()
    history, name = model.train()
    gh.save_history(history, name)
    logging.info(f"Testing {str(model.name)}")
    #model.test()
    #gh.show_results(name)

def main():
    models = [VGG16_Model, InceptionResNetV2_Model]
    for m in models:
        tf.keras.backend.clear_session()
        forward_pass(m())


def testing():
    Base_Model().test(made_model="/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/results/ResNet50(0).keras")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
    #testing()