import yaml 
import sys
sys.path.append('utils')
sys.path.append('models')
import utils.file_helpers as fh
import utils.video_process as vp
import models.model1 as m
from models.model1 import Model
import tensorflow as tf
import os


with open("params\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)


def main():
    fh.split_data()
    print("Data has been split")
    m = Model()
    m.training_data = vp.extract_data('training')
    print(m.training_data)
    m.validation_data = vp.extract_data('validation')
    print("Data stacks acquired")
    m.train()
    print("Model trained")
    fh.return_data()

if __name__ == "__main__":
    main()