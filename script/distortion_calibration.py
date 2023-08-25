import cv2
import numpy as np
import os
import glob
import datetime
import pathlib
import pkl

import click

from br2_vision.utility import config_logging, get_script_logger
from br2_vision.naming import *


@click.command()
@click.option(
    "--directory",
    default="calibration_images",
    help="Directory where the calibration images are stored.",
)
@click.option(
    "--extension", default=".png", help='Extension of the image files (default=".png")'
)
@click.option("-o", "--output", type=click.Path(exists=False), help="Output file name.")
@click.option("--verbose", is_flag=True, help="Verbose mode")
def calibrate(directory: str, extension: str, output: pathlib.Path, verbose: bool):
    """
    Calibrate camera using checkerboard reference images in a given directory.

    Parameters
    ----------
    directory : str
        Directory where the calibration images are stored.
    extension : str
        Extension of the image files (default='.png')
    """

    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    # Calibration Criteria
    checkerboard = (6, 7)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    )

    # Define the number of object detecting on board
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : checkerboard[0], 0 : checkerboard[1]].T.reshape(-1, 2)

    # Use all images to do calibration
    _img_shape = None
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(directory + "/*" + extension)
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray[gray > _th] = 240
        gray[gray < _th] = 0

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            checkerboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), subpix_criteria
            )
            imgpoints.append(corners2)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    logger.info("# Calibrated: ", datetime.datetime.now())
    logger.info("# Found " + str(N_OK) + " valid images for calibration")
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )
    logger.info("DIM=" + str(_img_shape[::-1]))
    logger.info("K=np.array(" + str(K.tolist()) + ")")
    logger.info("D=np.array(" + str(D.tolist()) + ")")
    logger.info("rms: ", rms)

    # Save output to pkl file
    with open(output, "wb") as f:
        pkl.dump({"DIM": _img_shape[::-1], "K": K, "D": D}, f)


if __name__ == "__main__":
    calibrate("calibration_images")
