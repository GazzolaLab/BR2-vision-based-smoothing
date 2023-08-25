import cv2
import sys

import numpy as np
import matplotlib.pyplot as plt

import os
import glob

from config import *

"""
The video includes LED light that indicates the active status of the experiment.
The script asks for a region-of-interest (roi), which selects where the LED is
visible, and trim the video for each interval that LED light is on.
"""

def process(camera_id:int, fps=60, run_id=1,
        trailing_frames=0, led_threshold=(70,65,150)):
    """
    Trimming process. Script asks for ROI and trim the video.

    Parameters
    ----------
    camera_id : int
        Camera index given in file.
    fps : int
        Frames per seconds (default=60)
    run_id : int
        Run index given in file
    trailing_frames : int
        Number of trailing frames after the LED status is turned off. (default=0)
    led_threshold : tuple(int,int,int)
        RGB threshold of the LED: greater value will be considered as 'on'
    """

    # Utility lambdas
    frame2timestr = lambda frame: str(frame/fps)
    
    # Path Configuration
    video_path = RAW_VIDEO_PATH.format(camera_id)
    output_path = RAW_FOOTAGE_VIDEO_PATH

    # Load Video
    capture = cv2.VideoCapture(video_path)

    # Select LED region
    r = cv2.selectROI('select roi', frame)
    print('LED region: ', r)
    
    # LED Threshold
    led_threshold = np.array(led_threshold)
    led_state = lambda c: np.linalg.norm(c) > np.linalg.norm(led_threshold)
    
    # Iterate Video
    current_state = False
    frame_count = 0
    while capture.isOpened():
        ret, frame = capture.read()
        frame_count += 1
        if frame is None:
            break

        # Crop Image
        imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        # Determine State
        average_color = imCrop.mean(axis=0).mean(axis=0)
        new_state = led_state(average_color)

        if new_state and not current_state:
            # Mark the start-frame
            start_frame = frame_count
            current_state = True
        elif not new_state and current_state:
            # Mark the end-frame
            end_frame = frame_count + trailing_frames
            # Trim
            start_stamp = frame2timestr(start_frame)
            end_stamp = frame2timestr(end_frame)
            current_state = False
            command = f'ffmpeg -y -i {video_path} -ss {start_stamp} \
                    -to {end_stamp} {output_path.format(camera_id, run_id)}'
            print(command)
            os.system(command)

            run_id += 1
    capture.release()

if __name__ == '__main__':
    for camera_id in range(1,6):
        run_id = 1
        process(camera_id=camera_id, run_id=run_id)
