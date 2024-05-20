import os
import random
import yaml
import logging
import sys
sys.path.append('params')

with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

def delete_file_type(path, filetype):
    """
    Deletes files of type filetype in path

    Inputs:
    path - paths.yaml header - validation
    filetype - in the format .filetype - .txt
    """
    data = os.listdir(params.get(path))
    for file in data:
        if file.endswith(filetype):
            os.remove(os.path.join(params.get(path), file))

def split_data():
    """
    Splits data from 'Data' folder into 'Training' and 'Validation' folders based off of 'split'
    Params from params.yaml
    """
    split = model_params.get("training_split")
    origin_data = os.listdir(params['origin_path']['data']) 
    origin_annotations = os.listdir(params['origin_path']['annotations'])
    videos= []
    annotations = []
    for d in origin_data:
        videos.append(d)
    for a in origin_annotations:
        annotations.append(a)
        params/params.yaml
    zipped_data = list(zip(videos, annotations))
    random.shuffle(zipped_data)
    training_amount = round(len(zipped_data)*split/100)
    #Training Data
    for video, annotation in zipped_data[:training_amount]:
        os.rename(os.path.join(params['origin_path']['data'], video), os.path.join(params['training_path']['data'], video))
        os.rename(os.path.join(params['origin_path']['annotations'], annotation), os.path.join(params['training_path']['annotations'], annotation))
    #Validation data
    for video, annotation in zipped_data[training_amount:]:
        os.rename(os.path.join(params['origin_path']['data'], video), os.path.join(params['validation_path']['data'], video))
        os.rename(os.path.join(params['origin_path']['annotations'], annotation), os.path.join(params['validation_path']['annotations'], annotation))
    logging.info("  Data has been split")

def return_data():
    """
    Returns data from 'training_path'(s) and 'validation_path'(s)  to 'data_path' folder.
    Params from params.yaml
    """
    origin_to_end = {params['training_path']['data']: params['origin_path']['data'],
                     params['training_path']['annotations']: params['origin_path']['annotations'],
                     params['validation_path']['data']: params['origin_path']['data'],
                     params['validation_path']['annotations']: params['origin_path']['annotations']}
    for key in origin_to_end:
        folder = os.listdir(key)
        for file in folder:
            os.rename(os.path.join(key, file), os.path.join(origin_to_end[key], file))
    logging.info("  Returned data")

def setup():
    logging.info("Checking if directories are setup")
    paths = [params['origin_path'], params['training_path'], params['validation_path']]
    for path in paths:
        for folder_path in path.values():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                logging.info(f"Created folder: {folder_path}")
            else:
                logging.info(f"Folder already exists: {folder_path}")
    logging.info("Setup completed")
    
setup()
