import cv2
import yaml
import file_helpers

with open("params\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)


def get_frames(video_path):
    """
    Gets individual frames from video_path specified by 'frame_rate'.
    Params from params.yaml

    Inputs:
    video_path - path to video for frame extraction
    
    Returns:
    List of raw frames from video_path
    """
    frame_rate = params.get("frame_rate")
    frames = []
    count = 0
    cap = cv2.VideoCapture(video_path)
    flag = True
    while flag:
        flag, frame = cap.read()
        if flag: #If frame is returned
            if count % frame_rate == 0: #If correct frame_rate
                frames.append(frame)
                resize_frames(frames)
                break
        else:
            break
        count += 1
    cap.release()
    return frames

def resize_frames(frames):
    """
    Resizes all frames into 'model_width' and 'model_height' sizes
    Params from params.yaml

    Input:
    frames - A list of frames
    """
    n_w = params.get("model_width")
    n_h = params.get("model_height")
    colour_chanel = params.get("colour_chanel")
    #If model colour chanel needs to be RGB
    if colour_chanel == "RGB":
        frames = [cv2.resize(f[:,:,::-1], (n_h, n_w)) /255 for f in frames]
    else: # Else keep BGR
        frames = [cv2.resize(f, (n_h, n_w)) /255 for f in frames]
    return frames

