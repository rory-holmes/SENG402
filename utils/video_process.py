import cv2
import yaml
import tensorflow as tf
import os
with open("params\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

with open("params\model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

n_w = model_params.get("image_width")
n_h = model_params.get("image_height")
colour_chanel = params['settings']["colour_chanel"]
frame_rate = params['settings']["frame_rate"]

def extract_data(folder_path):
    """
    Extracts all frames and labels within paths provided and converts into tensors.
    Params from params.yaml

    Input:
    folder_path - Either 'training' or 'validation'

    Returns:
    dataset - TensorFlow dataset from tensor slices of frames and annotations
    """
    if folder_path == 'training':
        video_path = params['training_path']['data']
        annotation_path = params['training_path']['annotations']
    elif folder_path == 'validation':
        video_path = params['validation_path']['data']
        annotation_path = params['validation_path']['annotations']
    else:
        raise ValueError("Incorrect value for 'folder_path' must be 'training' or 'validation'")
    
    frame_stack = []
    for video in os.listdir(video_path):
        frames = get_frames(os.path.join(video_path, video))
        frame_stack.append(frames)
    frame_stack = tf.convert_to_tensor(frame_stack)

    annotation_stack = []
    for file in os.listdir(annotation_path):
        annotation = get_labels(os.path.join(annotation_path, file))
        annotation_stack.append(annotation)
    print(annotation_stack) #TODO remove this
    annotation_stack = tf.convert_to_tensor(annotation_stack)

    return tf.data.Dataset.from_tensor_slices((frame_stack, annotation_stack))

def get_labels(path):
    """
    Extracts all labels for a video. 
    Needs to be updated based on format of label file.
    
    Inputs:
    path - A path to the text file containing labels

    Returns:
    labels - A list of all class labels within a video
    """
    labels = []
    file = open(path, "r")
    #Label file contains a header
    for label in file.readlines()[1:]:
        #Labels are separated by \t
        label = label.split("\t")
        #Labels contain a \n at the end of the line
        label[-1] = label[-1][0]
        #Lables have a frame annotation
        labels.append(label[1:])
        break #TODO Remove this line
    return labels

def get_frames(video_path):
    """
    Gets individual frames from video_path specified by 'frame_rate'.
    Params from params.yaml

    Inputs:
    video_path - path to video for frame extraction

    Returns:
    List of raw frames from video_path
    """
    frames = []
    count = 0
    cap = cv2.VideoCapture(video_path)
    flag = True
    while flag:
        flag, frame = cap.read()
        if flag: #If frame is returned
            if count % frame_rate == 0: #If correct frame_rate
                frame = resize_frame(frame)
                frames.append(frame)
                break #TODO remove this
        else:
            break
        count += 1
    cap.release()
    return frames

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

def frame_generator(video_list):
    """
    Generator function to yield frames from a framestack of multiple videos
    
    Input:
    video_list - A list of all the frames within all the videos
    
    Yields:
    A single frame in sequential order
    """
    for video_frames in video_list:
        for frame in video_frames:
            yield frame

