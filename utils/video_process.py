import cv2
import yaml
import os
import logging
import glob
import numpy as np
import random
import openpyxl

with open(r"params\\paths.yaml", "r") as f:
    paths = yaml.load(f, Loader=yaml.SafeLoader)

with open(r"params\\feature_model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

with open(r"params\\phase_model_params.yaml", "r") as f:
    phase_model_params = yaml.load(f, Loader=yaml.SafeLoader)

n_w = model_params.get("image_width")
n_h = model_params.get("image_height")
colour_chanel = "BGR"
frame_rate = model_params["frame_rate"]
batch_size = model_params['batch_size']

# Feature Extraction ------------------------------------------------------------------------
def data_generator(folder_path, batch_size):
    """
    Extracts all frames and labels within paths provided and converts into tensors, 
    yields batches of frame, label pairs for training or validation.
    Params from paths.yaml

    Input:
    folder_path - Either 'training', 'validation', or 'test'
    batch_size - size of batches yielded

    Yield:
    (frames, annotations) length of batch_size
    """
    try:
        video_path = paths.get(f"{folder_path}_data")
        annotation_path = paths.get(f"{folder_path}_annotations")
    except Exception:
        raise ValueError("Incorrect value for 'folder_path' must be 'training', 'validation', or 'testing'")
    
    logging.info(f"\n  Data generator running for {folder_path}")
    
    data = list(zip(sorted(os.listdir(video_path)), sorted(os.listdir(annotation_path))))
    while True:
        random.shuffle(data)
        for video, file in data:
            frames_path = os.path.join(video_path, video)
            labels_path = os.path.join(annotation_path, file)
            for batch_frames, batch_labels in zip(frame_generator(frames_path, batch_size), label_generator(labels_path, batch_size)):
                if len(batch_frames) == len(batch_labels):
                    yield (np.array(batch_frames), np.array(batch_labels))

        if folder_path == "testing":
            break

def label_generator(path, batch_size):
    """
    Extracts all labels for a video, yields by batch_size. 
    
    Inputs:
    path - A path to the text file containing labels
    batch_size - quantity of labels to return per yield

    Yields:
    labels - A list of all class labels within a video
        
    Note:
    Would need to be updated based on format of label file.
    """
    file = open(path, "r")
    labels = []
    #Label file contains a header
    for label in file.readlines()[1:]:
        #Labels are separated by \t
        label = label.split("\t")
        #Labels contain a \n at the end of the line
        label[-1] = label[-1][0]
        #Lables have a frame annotation
        label = [int(l) for l in label[1:]]
        labels.append(label)
        if len(labels) == batch_size:
            yield labels
            labels = []

    if labels:
        yield labels

def frame_generator(video_path, batch_size):
    """
    Gets individual frames of length 'batch_size' from video_path specified by 'frame_rate'.
    Params from paths.yaml

    Inputs:
        video_path - path to video for frame extraction
        batch_size - quantity of frames to return

    Yields:
        List of raw frames from video_path of size batch_size
    """
    count = 0
    cap = cv2.VideoCapture(video_path)
    frames = []
    flag = True
    while flag:
        flag, frame = cap.read()
        if flag: #If frame is returned
            if count % frame_rate == 0: #If correct frame_rate
                frame = resize_frame(frame)
                frames.append(frame)
            if len(frames) == batch_size:
                yield frames
                frames = []
            count += 1
    cap.release()
    if frames:
        yield frames

def resize_frame(frame):
    """
    Resizes all frames into 'model_width' and 'model_height' sizes
    Params from model_params.yaml

    Input:
        frames - A list of frames
    """
    #If model colour chanel needs to be RGB
    if colour_chanel == "RGB":
        frame = cv2.resize(frame[:,:,::-1], (n_h, n_w)) /255
    else: # Else keep BGR
        frame = cv2.resize(frame, (n_h, n_w)) /255
    return frame

def get_steps(folder_path):
    """
    Gets the length of all files found within folder_path and divides by batch_size.
    Params from paths.yaml

    Inputs:
    folder_path - Path to the folder used to calculate length of files 

    Returns:
    Steps necessary based on folder_path size        
    """
    steps = 0
    for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            steps += len(lines)
    return steps//model_params['batch_size']

def get_training_validation_steps():
    """
    Returns training steps, validation steps for feature training
    """
    return (get_steps(paths['training_annotations']), get_steps(paths['validation_annotations']))

# Phase detection --------------------------------------------------------------------

def phase_generator(stage):
    """
    Extracts all frames and labels within paths provided and converts into tensors, 
    yields batches of frame, label pairs for training or validation.
    Params from paths.yaml

    Input:
        stage - Either 'training', 'validation', or 'test'

    Yield:
        (frames, annotations) length of batch_size
    """
    logging.info(f"\n  Data generator running for {path}")

    try:
        path = paths.get(f"{stage}_data")
    except Exception:
        raise ValueError("Incorrect value for 'stage' must be 'training', 'validation', or 'testing'")
    
    data = os.listdir(path)
    while True:
        random.shuffle(data)
        for video in data:
            for batch in phase_video_generator(video, path):
                yield batch

def phase_video_generator(video_name, path):
    """
    Yields batches of frames and labels for the given video
    """
    if phase_data:
        phase_data = extract_phase(video_name)
        vid_path = os.path.join(path, video_name)
        frame_index = 0
        for batch_frames in frame_generator(vid_path, batch_size*model_params['stack_size']):
            batch_labels = [get_current_phase(frame_index + (i*25), phase_data) for i in range(batch_size*model_params['stack_size'])]
            frame_index += batch_size*25*model_params['stack_size']
            if len(batch_frames) == len(batch_labels):
                yield (np.array(batch_frames), np.array(batch_labels))

def extract_phase(video_name):
    """
    Extracts video phase data in seconds from Colorectal Annotations document based on video name
    Inputs:
        video_name: name of the video to retieve corresponding labels
    Returns:
        Phase labels in the format:
            [Time retaction start,   Time dissection start,   Time vessel ligated,	  Completing dissection]
    """
    phase_data = None
    path = os.path.join(paths['phase_annotations_path'], r"Colorectal-Annotations-V2.xlsx")
    obj = openpyxl.load_workbook(path)
    for row in obj.active.iter_rows(values_only=True): 
        if row[0] and row[0].strip() == video_name.split(".")[0]:
            phase_data = row
            break

    if not(phase_data):
        raise ValueError(f"Video name not found: {video_name}")
    
    if phase_data[2] == 1:
        return phase_data[8:16:2]
    
def get_current_phase(current_frame, phase_data):
    """
    Returns a one-hot encoding of what the current frame is on based on frame index
    """
    one_hot = [0 for _ in range(len(phase_data))]
    current_second = current_frame/phase_model_params['frame_rate']
    for i in range(len(phase_data)-1, -1, -1):
        if current_second >= phase_data[i]:
            one_hot[i] = 1

    if i == 0:
        one_hot[0] = 1
    return one_hot

def get_phase_steps(folder_path):
    """
    Counts the frames per frame rate of the videos in the specified dir

    Inputs:
        folder_path: Specified folder_path
    
    Returns:
        Steps found in the specified dir
    """
    steps = 0
    for video_name in glob.glob(os.path.join(folder_path, '*.mpg')):
        video = cv2.VideoCapture(video_name)
        steps += video.get(cv2.CAP_PROP_FRAME_COUNT)/phase_model_params['frame_rate']
    return steps//model_params['batch_size']

def get_phase_training_validation_steps():
    """
    Returns training steps, validations steps for phase training
    """
    return get_phase_steps(paths['training_data']), get_phase_steps(paths['validation_data'])



    
