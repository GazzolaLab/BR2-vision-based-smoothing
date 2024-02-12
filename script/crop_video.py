__doc__ = """
Take synchronized video path, crop the video and save it in cropped_video_path.
Use cv2 to select roi for each video.
Use ffmpeg to crop the video.
"""

from typing import Tuple

import cv2
import subprocess
import sys
import pathlib
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import os
import glob

import click

import br2_vision
from br2_vision.utility.logging import config_logging, get_script_logger
from br2_vision.cv2_custom.utils import select_roi


def crop_video(
    position: Tuple[int, int],
    section: Tuple[int, int],
    input_path: str,
    output_path: str,
):
    """
    Crop video using ffmpeg.
    position relative to top(y) left(x) corner.
        (x,y): (distance from left, distance from top)
    section (width-delta x, height-delta y)
    Assume position and section are in pixels.
    """
    width, height = section
    x, y = position

    command = ["ffmpeg", "-y"]
    command.extend(["-i", input_path])
    command.extend(["-vf", f"crop={width}:{height}:{x}:{y}"])
    command.extend([output_path])
    command = " ".join(command)
    print("running : ", command)

    sts = subprocess.Popen(command, shell=True).wait()
    return sts


@click.command()
@click.option(
    "-c", "--cam-id", type=int, help="Camera index given in file.", multiple=True
)
@click.option(
    "-skip-synch", is_flag=True, type=bool, help="Skip synchronization step. Use raw."
)
@click.option(
    "-roi", type=(int, int, int, int), optional=True, help="Region of interest for cropping. (x,y,width,height). Used for all cameras."
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def process(cam_id, skip_synch: bool, roi, verbose: bool, dry: bool):
    """
    Crop video using ffmpeg.
    """
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    # Path Configuration
    video_path = (
        config["PATHS"]["raw_video_path"]
        if skip_synch
        else config["PATHS"]["synchronized_video_path"]
    )
    output_path = config["PATHS"]["cropped_video_path"]

    # Select roi for each video
    if roi is None:
        rois = []
        for i in cam_id:
            assert os.path.exists(video_path.format(i))
            _input_path = video_path.format(i)

            r = select_roi(_input_path)
            if r is None:
                logger.error(f"Error selecting roi for camera {i}")
            rois.append(r)
    else:
        rois = [roi for _ in cam_id]

    # Crop video
    for i in tqdm(cam_id):
        roi = rois[i]
        _input_path = video_path.format(i)
        _output_path = output_path.format(i)
        positions = (roi[0], roi[1])
        section = (roi[2], roi[3])
        crop_video(positions, section, _input_path, _output_path)


if __name__ == "__main__":
    process()
