import os
import os
import random
import yaml
with open("params\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

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
    split = params.get("training_split")
    data = os.listdir(params.get("data_path")) 
    videos= []
    annotations = []
    for d in data:
        if d.endswith(params.get("data_type")):
            videos.append(d)
        elif d.endswith(params.get("annotation_type")):
            annotations.append(d)
        
    zipped_data = list(zip(videos, annotations))
    random.shuffle(zipped_data)
    training_amount = round(len(zipped_data)*split/100)
    #Training Data
    for video, annotation in zipped_data[:training_amount]:
        os.rename(os.path.join(params.get("data_path"), video), os.path.join(params.get("training_path"), video))
        os.rename(os.path.join(params.get("data_path"), annotation), os.path.join(params.get("training_path"), annotation))
    #Validation data
    for video, annotation in zipped_data[training_amount:]:
        os.rename(os.path.join(params.get("data_path"), video), os.path.join(params.get("validation_path"), video))
        os.rename(os.path.join(params.get("data_path"), annotation), os.path.join(params.get("validation_path"), annotation))

def return_data():
    """
    Returns data from 'training_path' and 'validation_path'  to 'data_path' folder.
    Params from params.yaml
    """
    training = os.listdir(params.get("training_path"))
    validation = os.listdir(params.get("validation_path"))
    for d in training:
        os.rename(os.path.join(params.get("training_path"), d), os.path.join(params.get("data_path"), d))
    for d in validation:
        os.rename(os.path.join(params.get("validation_path"), d), os.path.join(params.get("data_path"), d))

