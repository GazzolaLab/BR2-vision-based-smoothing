import os

# import tensorflow as tf
import pdb
import re
import sys
from typing import List

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

import br2_vision
from br2_vision.data_structure import FlowQueue, MarkerPositions, TrackingData
from br2_vision.manual_tracing import ManualTracing
from br2_vision.optical_flow import CameraOpticalFlow
from br2_vision.utility.logging import config_logging, get_script_logger

stdin, stdout = sys.stdin, sys.stdout


def set_trace():
    pdb.Pdb(stdin=stdin, stdout=stdout).set_trace()


@click.command()
@click.option(
    "-t",
    "--tag",
    type=str,
    help="Experiment tag. Path ./tag should exist.",
)
@click.option(
    "-c",
    "--cam-id",
    type=int,
    help="Specify camera index.",
    required=True,
)
@click.option(
    "-r",
    "--run-id",
    type=int,
    help="Specify run index.",
    required=True,
)
@click.option(
    "--force-list-done-queue",
    is_flag=True,
    type=bool,
    help="If set, prompt all queues to be tracked.",
    default=False,
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def main(tag, cam_id, run_id, verbose, dry, force_list_done_queue):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))
    scale = float(config["DIMENSION"]["scale_video"])

    # Run optical flow for each run-id
    datapath = config["PATHS"]["tracing_data_path"].format(tag, run_id)
    with TrackingData.load(path=datapath) as dataset:

        # Render tracking video
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

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
