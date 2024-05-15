from typing import List
import os, sys

# import tensorflow as tf

import cv2
import matplotlib.pyplot as plt
import numpy as np

import br2_vision
from br2_vision.utility.logging import config_logging, get_script_logger
from br2_vision.data_structure import MarkerPositions, TrackingData, FlowQueue
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
    "-r",
    "--run-id",
    type=int,
    help="Specify run index. Initial points are saved for all specified run-ids.",
    multiple=True,
)
#@click.option(
#    "--force-run-all",
#    is_flag=True,
#    type=bool,
#    help="Ignore the pre-run data, and re-run optical flow on all flow-queues.",
#    default=False,
#)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def main(tag, run_id, verbose, dry):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))
    scale = float(config["DIMENSION"]["scale_video"])

    # Run optical flow for each run-id
    for rid in run_id:
        datapath = config["PATHS"]["tracing_data_path"].format(tag, rid)
        with TrackingData.load(path=datapath) as dataset:
            for cid in dataset.iter_cameras():
                queues: List[FlowQueue] = dataset.get_flow_queues(
                    camera=cid, 
                )
                print(f"camera: {cid} | Optical flow run on:")
                [print("    ", q) for q in queues]

                video_path = config["PATHS"]["footage_video_path"].format(tag, cid, rid)

                optical_flow = CameraOpticalFlow(
                    video_path=video_path,
                    flow_queues=queues,
                    dataset=dataset,
                    scale=scale,
                )
                optical_flow.run(debug=dry)

                all_queues = dataset.get_flow_queues(camera=cid, force_run_all=True)
                tracking_overlay_video_path = config["PATHS"][
                    "footage_video_path_with_trace"
                ].format(tag, cid, rid)
                optical_flow.render_tracking_video(
                    tracking_overlay_video_path, all_queues
                )

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
