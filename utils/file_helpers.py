import os
import random
import yaml
import logging
import sys
sys.path.append('params')

with open(r"params\\paths.yaml", "r") as f:
    paths = yaml.load(f, Loader=yaml.SafeLoader)

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
    data = os.listdir(paths.get(path))
    for file in data:
        if file.endswith(filetype):
            os.remove(os.path.join(paths.get(path), file))

def split_data():
    """
    Splits feature extraction data from 'Data' folder into 'Training' and 'Validation' folders based off of 'split'
    Params from paths.yaml
    """
    return_data() 
    split = feature_model.get("training_split")
    videos = sorted(os.listdir(paths['origin_data']))
    annotations = sorted(os.listdir(paths['origin_annotations']))

    zipped_data = list(zip(videos, annotations))
    random.shuffle(zipped_data)
    training_amount = round(len(zipped_data)*split/100)
    #Training Data
    for video, annotation in zipped_data[:training_amount]:
        os.rename(os.path.join(paths['origin_data'], video), os.path.join(paths['training_data'], video))
        os.rename(os.path.join(paths['origin_annotations'], annotation), os.path.join(paths['training_annotations'], annotation))
        
    #Validation data
    for video, annotation in zipped_data[training_amount:]:
        os.rename(os.path.join(paths['origin_data'], video), os.path.join(paths['validation_data'], video))
        os.rename(os.path.join(paths['origin_annotations'], annotation), os.path.join(paths['validation_annotations'], annotation))
        
    logging.info("  Data has been split")

def split_phase_data():
    """
    Splits phase data from 'Data' folder into 'Training' and 'Validation' folders based off of 'split'
    Params from paths.yaml
    """
    return_data(phase=True) 
    split = phase_model.get("training_split")
    origin_data = sorted(os.listdir(paths['phase_videos_path']))

    random.shuffle(origin_data)
    training_amount = round(len(origin_data)*split/100)
    #Training Data
    for video in origin_data[:training_amount]:
        os.rename(os.path.join(paths['phase_videos_path'], video), os.path.join(paths['training_data'], video))
        
    #Validation data
    for video in origin_data[training_amount:]:
        os.rename(os.path.join(paths['phase_videos_path'], video), os.path.join(paths['validation_data'], video))
        
    logging.info("  Data has been split")

def return_data(phase=False):
    """
    Returns data from 'training_path'(s) and 'validation_path'(s)  to 'data_path' folder.
    Inputs:
        phase: If phase is true returns phase data

    Params from params.yaml
    """
    # If returning phase data
    if phase == True:
        logging.info("Returning phase data")
        origin_to_end = {paths['training_data']: paths['phase_videos_path'],
                        paths['validation_data']: paths['phase_videos_path']}
    else:
        logging.info("Returning feature data")
        origin_to_end = {paths['training_data']: paths['origin_data'],
                        paths['training_annotations']: paths['origin_annotations'],
                        paths['validation_data']: paths['origin_data'],
                        paths['validation_annotations']: paths['origin_annotations']}
        
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

    for folder_path in paths['paths'].values():
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logging.info(f" Created folder: {folder_path}")
        else:
            logging.info(f" Folder already exists: {folder_path}")
    logging.info("Setup completed")

