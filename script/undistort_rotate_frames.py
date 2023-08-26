import cv2
import glob
import os, sys
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
@click.option("-f", "--file", type=click.Path(exists=True), help="Path to the calibration file")
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enables verbose mode."
)
def undistort_and_rotate_frames(cam_id, rotate, file, verbose):
    """
    Undistort and rotate calibration frames.

    Take the image collection from CALIBRATION_IMAGE_COLLECTION and apply undistortion and rotation.
    Note: The processed image is overwritten.
    """

    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    if rotate is not None:
        cv2_rotation = getattr(cv2, rotate, None)
    else:
        cv2_rotation = None

    image_path_collection = CALIBRATION_IMAGE_COLLECTION.format(camid)
    raw_images = glob.glob(image_path_collection, recursive=True)

    for path in raw_images:
        frame = cv2.imread(path)

        # Undistort
        height, width = frame.shape[:2]
        frame = undistort(frame, width, height, file)

        # Rotate
        if cv2_rotation != None:
            frame = cv2.rotate(frame, cv2_rotation)

        # Write
        cv2.imwrite(path, frame)

        logger.info("The image was successfully saved - {}".format(path))


if __name__ == "__main__":
    undistort_and_rotate_frames()
