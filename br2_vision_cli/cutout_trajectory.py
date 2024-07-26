import os
import re
import sys

import click
import numpy as np
import pandas as pd

import br2_vision
from br2_vision.cv2_custom.extract_info import get_video_frame_count
from br2_vision.data_structure import FlowQueue, MarkerPositions, TrackingData
from br2_vision.utility.logging import config_logging, get_script_logger


config = br2_vision.load_config()
config_logging(False)
logger = get_script_logger(os.path.basename(__file__))


@click.command()
@click.option(
    "-t",
    "--tag",
    type=str,
    required=True,
    help="Experiment tag.",
)
@click.option(
    "-r",
    "--run-id",
    type=int,
    required=True,
    help="Specify run index..",
)
@click.option(
    "-f",
    "--frame",
    type=int,
    required=True,
    help="Specify frame index.",
)
@click.option(
    "-z",
    "--z-index",
    type=int,
    required=True,
    help="Specify z-index.",
)
@click.option(
    "-l",
    "--label",
    type=str,
    required=True,
    help="Specify label.",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def main(tag, run_id, frame, z_index, label, verbose, dry):
    from br2_vision.data_structure.tracking_data import compose_tag

    q_tag = compose_tag(z_index, label)
    datapath = config["PATHS"]["tracing_data_path"].format(tag, run_id)
    with TrackingData.load(path=datapath) as dataset:
        dataset.trim_trajectory(q_tag, frame)
