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


def extract_audio_wav(input_path:str, output_path:str):
    command = ['ffmpeg', '-y']
    command.extend(['-i', input_path])
    command.extend(['-c:a:0', 'pcm_s16le'])
    command.extend([output_path])
    command = ' '.join(command)
    print('running : ', command)

    sts = subprocess.Popen(command, shell=True).wait()
    return sts


@click.command()
@click.option(
    "-c", "--cam-id", type=int, help="Camera index given in file.", multiple=True
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def process(
    cam_id, verbose: bool, dry: bool
):
    """
    Synchronize video

    Read all video in config["PATHS"]["raw_video_path"] and create
    config["PATHS"]["synchronized_video_path"]

    Parameters
    ----------
    path: str
        Video path
    cam_id : int
        Camera index given in file (multiple).
    """
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))
    path = config["PATHS"]["data_dir"]

    # Path Configuration
    video_path = config["PATHS"]["raw_video_path"]
    sync_video_path = config["PATHS"]["synchronized_video_path"]
    audio_path = config["PATHS"]["synchronized_audio_path"]

    for i in tqdm(cam_id):
        assert os.path.exists(video_path.format(i))
        extract_audio_wav(
            video_path.format(i), audio_path.format(i))

    


if __name__ == "__main__":
    process()
