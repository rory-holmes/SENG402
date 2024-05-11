import yaml 
import os
import sys
sys.path.append('utils')
import utils.file_helpers as fh
import utils.video_process as vp

with open("params\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

def extract_frames(path):
    """
    Extracts all frames from 'data_type' files in 'data_path'. 
    Params from params.yaml

    Input:
    path - path to folder cotnaining data for frame collection

    Returns:
    frame_stack - stack of all frames from data_path
    """
    frame_stack = []
    for d in os.listdir(path):
        if d.endswith(params.get("data_type")):
            frames = vp.get_frames(os.path.join(path, d))
            frame_stack.append(frames)
    return frame_stack

def main():
    fh.split_data()
    frame_stack = extract_frames(params.get("training_path"))
    fh.return_data()
    print(len(frame_stack))


if __name__ == "__main__":
    main()