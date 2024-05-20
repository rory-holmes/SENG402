import yaml 
import sys
sys.path.append('utils')
sys.path.append('models')
import utils.file_helpers as fh
import utils.video_process as vp
import utils.global_helpers as gp
from models.model1 import Model, Sequential_Model
import tensorflow as tf
import os
import logging

with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

def forward_pass(model):
    fh.split_data()
    history = model.train(vp.extract_data)
    fh.return_data()
    print(gp.show_results(history))

def main():
    logging.getLogger().setLevel(logging.INFO)
    forward_pass(Sequential_Model())

if __name__ == "__main__":
    main()