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
    "-c",
    "--cam-id",
    type=int,
    required=True,
    help="camera index",
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
def main(tag, cam_id, run_id, frame, z_index, label, verbose, dry):
    from br2_vision.data_structure.tracking_data import compose_tag

    q_tag = compose_tag(z_index, label)
    datapath = config["PATHS"]["tracing_data_path"].format(tag, run_id)
    with TrackingData.load(path=datapath) as dataset:
        dataset.trim_trajectory(q_tag, frame, cam_id)

        if verbose:
            from br2_vision.optical_flow import CameraOpticalFlow
            scale = float(config["DIMENSION"]["scale_video"])
            video_path = config["PATHS"]["footage_video_path"].format(tag, cam_id, run_id)
            all_queues = dataset.get_flow_queues(camera=cam_id, force_run_all=True)
            optical_flow = CameraOpticalFlow(
                video_path=video_path,
                flow_queues=all_queues,
                dataset=dataset,
                scale=scale,
            )
            tracking_overlay_video_path = config["PATHS"][
                "footage_video_path_with_trace"
            ].format(tag, cam_id, run_id)

            optical_flow.render_tracking_video(tracking_overlay_video_path, cam_id)

