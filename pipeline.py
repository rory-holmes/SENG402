import yaml 
import sys
sys.path.append('utils')
sys.path.append('models')
import utils.file_helpers as fh
import utils.video_process as vp
from models.model1 import Model, Sequential_Model
import tensorflow as tf
import os
import logging

with open("params\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

def forward_pass(model):
    fh.split_data()
    model.training_data = vp.extract_data('training')
    print(model.training_data)
    model.validation_data = vp.extract_data('validation')
    evaluation = model.train()
    print(evaluation)
    fh.return_data()

def main():
    logging.getLogger().setLevel(logging.INFO)
    forward_pass(Sequential_Model())

if __name__ == "__main__":
    main()