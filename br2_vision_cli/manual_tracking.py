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
@click.option(
    "--iterate",
    is_flag=True,
    type=bool,
    help="Iter mode. Don't use with force-list-done-queue",
    default=False,
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def main(tag, run_id, verbose, dry, force_list_done_queue, iterate):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))
    scale = float(config["DIMENSION"]["scale_video"])

    # Run optical flow for each run-id
    datapath = config["PATHS"]["tracing_data_path"].format(tag, run_id)
    with TrackingData.load(path=datapath) as dataset:
        queues: list[FlowQueue] = dataset.get_flow_queues(
            force_run_all=force_list_done_queue,
        )
        if len(queues) == 0:
            logger.info(
                f"No flow-queue found. Please select a queue using set_optical_flow_inquiry."
            )
            return

        # Print all queues
        for idx, q in enumerate(queues):
            print(f"Queue {idx}: {q}")

        if iterate:
            tasks = list(range(len(queues)))
        else:
            # Prompt user to select a queue
            queue_idx = input("Select a queue:")

            # Check if the input is valid, using regex
            if not re.match(r"^\d+$", queue_idx):
                logger.error("Invalid input. Please input a number.")
                return
            queue_idx = int(queue_idx)
            # check if the input is within the range
            if queue_idx < 0 or queue_idx >= len(queues):
                logger.error("Invalid input. Please input a number within the range.")
                return
            tasks = [queue_idx]

        for queue_idx in tasks:
            queue = queues[queue_idx]
            print("working on: ")
            print(queue)
            cid = queue.camera

            video_path = config["PATHS"]["footage_video_path"].format(tag, cid, run_id)

            tracing = ManualTracing(
                video_path=video_path,
                flow_queue=queue,
                dataset=dataset,
                scale=scale,
            )
            # set_trace()
            result = tracing.run(debug=dry)
            if not result:
                logger.info(f"Failed to trace queue {queue_idx}.")
                return

            if verbose:
                # Render tracking video
                all_queues = dataset.get_flow_queues(camera=cid, force_run_all=True)
                optical_flow = CameraOpticalFlow(
                    video_path=video_path,
                    flow_queues=all_queues,
                    dataset=dataset,
                    scale=scale,
                )
                tracking_overlay_video_path = config["PATHS"][
                    "footage_video_path_with_trace"
                ].format(tag, cid, run_id)

                optical_flow.render_tracking_video(tracking_overlay_video_path, cid)

                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
