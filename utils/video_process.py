import cv2
import yaml
import tensorflow as tf
import os
import logging
import numpy as np
with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

with open("/csse/users/rho66/Desktop/Years/4/SENG402/SENG402/params/model_params.yaml", "r") as f:
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
    iterations = 1
    if folder_path == 'training':
        video_path = params['training_path']['data']
        annotation_path = params['training_path']['annotations']
        iterations = model_params['epochs']
    elif folder_path == 'validation':
        video_path = params['validation_path']['data']
        annotation_path = params['validation_path']['annotations']
    elif folder_path == 'testing':
        video_path = params['testing_path']['data']
        annotation_path = params['testing_path']['annotations']
    else:
        raise ValueError("Incorrect value for 'folder_path' must be 'training', 'validation', or 'testing'")
    
    for _ in range(iterations):
        for video, file in zip(sorted(os.listdir(video_path)), sorted(os.listdir(annotation_path))):
            logging.info(f"\n  Extracting frames from {video} and {file}")
            frames_path = os.path.join(video_path, video)
            labels_path = os.path.join(annotation_path, file)
            for batch_frames, batch_labels in zip(frame_generator(frames_path, batch_size), label_generator(labels_path, batch_size)):
                #logging.info(f"{i}/{len(frames)}")
                if len(batch_frames) == len(batch_labels):
                    yield (np.array(batch_frames), np.array(batch_labels))

def label_generator(path, batch_size):
    """
    Extracts all labels for a video. 
    Needs to be updated based on format of label file.
    
    Inputs:
    path - A path to the text file containing labels

    Returns:
    labels - A list of all class labels within a video
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
    Gets individual frames from video_path specified by 'frame_rate'.
    Params from params.yaml

    Inputs:
    video_path - path to video for frame extraction

    Returns:
    List of raw frames from video_path
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
    Params from params.yaml

    Input:
    frames - A list of frames
    """
    #If model colour chanel needs to be RGB
    if colour_chanel == "RGB":
        frame = cv2.resize(frame[:,:,::-1], (n_h, n_w)) /255
    else: # Else keep BGR
        frame = cv2.resize(frame, (n_h, n_w)) /255
    return frame

