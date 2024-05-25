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

with open("params\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

def forward_pass(model):
    fh.split_data()
    history, name = model.train(vp.data_generator)
    gh.save_history(history, name)
    fh.return_data()
    gh.show_results(name)

def main():
    logging.getLogger().setLevel(logging.INFO)
    forward_pass(ResNet50_Model())
    forward_pass(InceptionResNetV2_Model())
    forward_pass(VGG16_Model())
    
if __name__ == "__main__":
    main()