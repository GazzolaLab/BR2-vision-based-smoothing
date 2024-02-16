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


def export(input_path, output_path, start_stamp, end_stamp):
    command = ["ffmpeg", "-y"]
    command.extend(["-i", input_path])
    command.extend(["-ss", start_stamp])
    command.extend(["-to", end_stamp])
    command.extend([output_path])
    command = " ".join(command)
    print("running : ", command)

    sts = subprocess.Popen(command, shell=True).wait()
    return sts
    # TODO: dont use os.system, use subsystem
    command = f"ffmpeg -y -i {input_path} -ss {start_stamp} \
            -to {end_stamp} {output_path}"
    os.system(command)


@click.command()
@click.option(
    "-t",
    "--tag",
    type=str,
    help="Experiment tag. Path ./tag should exist.",
)
@click.option(
    "-c", "--cam-id", type=int, help="Camera index given in file.", multiple=True
)
@click.option("-ss", type=str, help="Start stamp", multiple=True)
@click.option("-to", type=str, help="End stamp", multiple=True)
@click.option(
    "-f",
    "--fps",
    type=int,
    default=60,
    help="Frames per seconds (default=60). Make sure it matches the orignal framerate of the video.",
)
@click.option(
    "-tr",
    "--trailing-frames",
    type=int,
    default=0,
    help="Number of trailing frames after the LED status is turned off. (default=0)",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def process(tag, cam_id, ss, to, fps, trailing_frames, verbose: bool, dry: bool):
    """
    Trimming video given start and end stamps.

    The video includes LED light that indicates the active status of the experiment.
    The script asks for a region-of-interest (roi), which selects where the LED is
    visible, and trim the video for each interval that LED light is on.

    Parameters
    ----------
    tag : pathlib.Path
        Experiment tag. The raw data are expected to be stored in the path ./tag
    cam_id : int
        Camera index given in file (multiple).
    fps : int
        Frames per seconds (default=60)
    run_id : int
        Run index given in file
    trailing_frames : int
        Number of trailing frames after the LED status is turned off. (default=0)
    led_threshold : tuple(int,int,int)
        RGB threshold of the LED: greater value will be considered as 'on'
    """
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    # Utility lambdas
    frame2timestr = lambda frame: str(frame / fps)

    # Path Configuration
    os.makedirs(config["PATHS"]["postprocessing_path"].format(tag), exist_ok=True)
    video_path = config["PATHS"]["preprocessed_video_path"]  # (cam_id)
    output_path = config["PATHS"]["footage_video_path"]  # (tag, cam_id, run_id)

    # Select LED regions for all cameras
    run_id = 0
    for cid in cam_id:
        for _ss, _to in zip(ss, to):
            video_path_ = video_path.format(cid)
            output_path_ = output_path.format(tag, cid, run_id)
            logger.info(f"Processing camera {cid}: {video_path_}, {_ss} to {_to}")
            if dry:
                continue
            export(
                input_path=video_path_,
                output_path=output_path_,
                start_stamp=_ss,
                end_stamp=_to,
            )
            run_id += 1


if __name__ == "__main__":
    process()
