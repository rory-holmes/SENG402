import os
import random
import yaml
import logging
import sys
sys.path.append('params')

with open(r"params\\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

with open(r"params\\feature_model_params.yaml", "r") as f:
    feature_model = yaml.load(f, Loader=yaml.SafeLoader)

with open(r"params\\phase_model_params.yaml", "r") as f:
    phase_model = yaml.load(f, Loader=yaml.SafeLoader)

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
    return_data() 
    split = feature_model.get("training_split")
    origin_data = sorted(os.listdir(params['origin_data']))
    origin_annotations = sorted(os.listdir(params['origin_annotations']))
    videos= []
    annotations = []
    for d in origin_data:
        videos.append(d)
    for a in origin_annotations:
        annotations.append(a)

    zipped_data = list(zip(videos, annotations))
    random.shuffle(zipped_data)
    training_amount = round(len(zipped_data)*split/100)
    #Training Data
    for video, annotation in zipped_data[:training_amount]:
        os.rename(os.path.join(params['origin_data'], video), os.path.join(params['training_data'], video))
        os.rename(os.path.join(params['origin_annotations'], annotation), os.path.join(params['training_annotations'], annotation))
        
    #Validation data
    for video, annotation in zipped_data[training_amount:]:
        os.rename(os.path.join(params['origin_data'], video), os.path.join(params['validation_data'], video))
        os.rename(os.path.join(params['origin_annotations'], annotation), os.path.join(params['validation_annotations'], annotation))
        
    logging.info("  Data has been split")

def split_phase_data():
    """
    Splits data from 'Data' folder into 'Training' and 'Validation' folders based off of 'split'
    Params from params.yaml
    """
    return_phase_data() 
    split = phase_model.get("training_split")
    origin_data = sorted(os.listdir(params['phase_videos_path']))

    random.shuffle(origin_data)
    training_amount = round(len(origin_data)*split/100)
    #Training Data
    for video in origin_data[:training_amount]:
        os.rename(os.path.join(params['phase_videos_path'], video), os.path.join(params['training_data'], video))
        
    #Validation data
    for video in origin_data[training_amount:]:
        os.rename(os.path.join(params['phase_videos_path'], video), os.path.join(params['validation_data'], video))
        
    logging.info("  Data has been split")

def return_phase_data():
    """
    Returns data from 'training_path'(s) and 'validation_path'(s)  to 'data_path' folder.
    Params from params.yaml
    """
    origin_to_end = {params['training_data']: params['phase_videos_path'],
                     params['validation_data']: params['phase_videos_path']}
    
    for key in origin_to_end:
        folder = os.listdir(key)
        for file in folder:
            os.rename(os.path.join(key, file), os.path.join(origin_to_end[key], file))
    logging.info("  Returned data")

def return_data():
    """
    Returns data from 'training_path'(s) and 'validation_path'(s)  to 'data_path' folder.
    Params from params.yaml
    """
    origin_to_end = {params['training_data']: params['origin_data'],
                     params['training_annotations']: params['origin_annotations'],
                     params['validation_data']: params['origin_data'],
                     params['validation_annotations']: params['origin_annotations']}
    for key in origin_to_end:
        folder = os.listdir(key)
        for file in folder:
            os.rename(os.path.join(key, file), os.path.join(origin_to_end[key], file))
    logging.info("  Returned data")

def setup():
    """
    Setups initial (data, training, validation, testing, results) directories
    """
    logging.info("Checking if directories are setup")

    for folder_path in params['paths'].values():
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logging.info(f" Created folder: {folder_path}")
        else:
            logging.info(f" Folder already exists: {folder_path}")
    logging.info("Setup completed")

