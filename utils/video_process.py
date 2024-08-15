import cv2
import yaml
import os
import logging
import glob
import numpy as np
import random

with open("params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

with open("params/model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

n_w = model_params.get("image_width")
n_h = model_params.get("image_height")
colour_chanel = params['settings']["colour_chanel"]
frame_rate = params['settings']["frame_rate"]
batch_size = model_params['batch_size']

def data_generator(folder_path, batch_size):
    """
    Extracts all frames and labels within paths provided and converts into tensors, 
    yields batches of frame, label pairs for training or validation.
    Params from params.yaml

    Input:
    folder_path - Either 'training', 'validation', or 'test'
    batch_size - size of batches yielded

    Yield:
    (frames, annotations) length of batch_size
    """
    if folder_path == 'training':
        video_path = params['training_path']['data']
        annotation_path = params['training_path']['annotations']
    elif folder_path == 'validation':
        video_path = params['validation_path']['data']
        annotation_path = params['validation_path']['annotations']
    elif folder_path == 'testing':
        video_path = params['testing_path']['data']
        annotation_path = params['testing_path']['annotations']
    else:
        raise ValueError("Incorrect value for 'folder_path' must be 'training', 'validation', or 'testing'")
    logging.info(f"\n  Data generator running for {folder_path}")
    
    data = zip(sorted(os.listdir(video_path)), sorted(os.listdir(annotation_path)))
    while True:
        for video, file in random.shuffle(data):
            #logging.info(f"\n  Extracting frames from {video} and {file}")
            frames_path = os.path.join(video_path, video)
            labels_path = os.path.join(annotation_path, file)
            for batch_frames, batch_labels in zip(frame_generator(frames_path, batch_size), label_generator(labels_path, batch_size)):
                #logging.info(f"{i}/{len(frames)}")
                if len(batch_frames) == len(batch_labels):
                    yield (np.array(batch_frames), np.array(batch_labels))

        #TODO Implement shuffling of training data
        if folder_path == "testing":
            break

def get_steps(folder_path):
    """
    Gets the length of all files found within folder_path and divides by batch_size.
    Params from params.yaml

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
    return steps/model_params['batch_size']

def get_training_validation_steps():
    """
    Returns training steps, validation steps
    """
    return (get_steps(params['training_path']['annotations']), get_steps(params['validation_path']['annotations']))

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
        #break #TODO Remove this line
        labels.append(label)
        if len(labels) == batch_size:
            yield labels
            labels = []

    if labels:
        yield labels

def frame_generator(video_path, batch_size):
    """
    Gets individual frames of length 'batch_size' from video_path specified by 'frame_rate'.
    Params from params.yaml

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
        else:
            break
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

