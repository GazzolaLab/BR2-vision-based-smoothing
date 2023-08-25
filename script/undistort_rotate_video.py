import os, sys
import pathlib
import logging
import cv2
import glob
import argparse

from .undistort import undistort

import click

from br2_vision.utility.logging import config_logging, get_script_logger
from br2_vision.naming import *


@click.command()
@click.option("-c", "--cam-id", type=int, help="Camera index")
@click.option(
    "-r",
    "--rotate",
    type=str,
    default=None,
    help="Rotation in cv2 (ex:ROTATE_90_CLOCKWISE)",
)
@click.option(
    "--output-fps",
    type=int,
    default=60,
    help="Output video FPS. Try to match the original video settings.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose")
def undistort_and_rotate(cam_id, rotate, output_fps, verbose):
    """
    Undistort and rotate the video.
    """

    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    if rotate is not None:
        cv2_rotation = getattr(cv2, rotate, None)
    else:
        cv2_rotation = None

    raw_videos = glob.glob(RAW_FOOTAGE_VIDEO_PATH_WILD.format(cam_id), recursive=True)

    for video_path in raw_videos:
        basename = os.path.basename(video_path)[:-4]
        save_path = os.path.join(PATH, "postprocess", basename + ".mp4")
        logger.info(f"{save_path=}")

        # Create an object to read
        video = cv2.VideoCapture(video_path)

        # We need to check if camera
        # is opened previously or not
        if video.isOpened() == False:
            logger.info("Error reading video file {}".format(video_path))
            continue

        # We need to set resolutions.
        # so, convert them from float to integer.
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))

        size = (frame_height, frame_width)  # Make sure the size is upright

        # Below VideoWriter object will create
        # a frame of above defined The output
        # is stored in 'filename.avi' file.
        result = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size
        )

        logger.info("writing video...")
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Undistort
            frame = undistort(frame)

            # Rotate
            if cv2_rotation != None:
                frame = cv2.rotate(frame, cv2_rotation)

            # Write video
            result.write(frame)

        # When everything done, release
        # the video capture and video
        # write objects
        video.release()
        result.release()

        logger.info("The video was successfully saved - {}".format(save_path))


if __name__ == "__main__":
    undistort_and_rotate()
