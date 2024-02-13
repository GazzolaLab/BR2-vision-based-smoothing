import os, sys
# import tensorflow as tf

import cv2
import matplotlib.pyplot as plt
import numpy as np

import br2_vision
from br2_vision.utility.logging import config_logging, get_script_logger
from br2_vision.data import MarkerPositions, TrackingData, FlowQueue
from br2_vision.optical_flow import CameraOpticalFlow

import click


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
@click.option(
    "-r", "--run-id", type=int, help="Specify run index. Initial points are saved for all specified run-ids.", multiple=True
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def main(tag, cam_id, run_id, verbose, dry):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    import argparse

    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", type=int, default=1)
    parser.add_argument("--camid", type=int, default=1)
    args = parser.parse_args()

    # Configuration: experiment setup
    RUNID = args.runid
    CAMID = args.camid



    # Initialization
    optical_flow = CameraOpticalFlow(
        camera_id=CAMID,
        data_path=TRACKING_FILE.format(CAMID, RUNID),
        video_path=PREPROCESSED_FOOTAGE_VIDEO_PATH.format(CAMID, RUNID),
        debug=True,
    )
    while not optical_flow.inquiry_empty():
        optical_flow.next_inquiry()
    optical_flow.save_data()
    optical_flow.save_tracking_video(
        PREPROCESSED_TRACKING_VIDEO_PATH.format(CAMID, RUNID), draw_label=None
    )

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
