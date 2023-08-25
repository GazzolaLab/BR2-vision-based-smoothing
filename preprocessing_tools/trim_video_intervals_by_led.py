import cv2
import sys

import numpy as np
import matplotlib.pyplot as plt

import os
import glob

import click

from utility.logging import config_logging, get_script_logger
from naming import *


@click.command()
@click.option('-c', '--cam-id', type=int, default=1, help='Camera index given in file.')
@click.option('-f', '--fps', type=int, default=60, help='Frames per seconds (default=60). Make sure it matches the orignal framerate of the video.')
@click.option('-r', '--run-id', type=int, default=1, help='Run index given in file')
@click.option('--trailing-frames', type=int, default=0, help='Number of trailing frames after the LED status is turned off. (default=0)')
@click.option('--led-threshold', type=(int,int,int), default=(70,65,150), help='RGB threshold of the LED: greater value will be considered as "on"')
def process(cam_id, fps, run_id, trailing_frames, led_threshold):
    """
    Trimming process. Script asks for ROI and trim the video.
    
    The video includes LED light that indicates the active status of the experiment.
    The script asks for a region-of-interest (roi), which selects where the LED is
    visible, and trim the video for each interval that LED light is on.

    Parameters
    ----------
    cam_id : int
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

    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    # Utility lambdas
    frame2timestr = lambda frame: str(frame/fps)
    
    # Path Configuration
    video_path = RAW_VIDEO_PATH.format(cam_id)
    output_path = RAW_FOOTAGE_VIDEO_PATH

    # Load Video
    capture = cv2.VideoCapture(video_path)

    # Select LED region
    r = cv2.selectROI('select roi', frame)
    logger.info('LED region: ', r)
    
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
                    -to {end_stamp} {output_path.format(cam_id, run_id)}'
            logger.info(command)
            os.system(command)

            run_id += 1
    capture.release()

if __name__ == '__main__':
    pricess()
    #for cam_id in range(1,6):
    #    run_id = 1
    #    process(cam_id=cam_id, run_id=run_id)
