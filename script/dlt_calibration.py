""" DLT Calibration Script

This script read the reference points and calculate the camera parameters for DLT method.
It assume the reference points are already selected for each camera. The calibration steps
are the following.

1. Import all the points in 3D-grid space.
2. Construct DLT module with the number of camera and reference points.
3. Finalize the DLT module

optional:
4. Compute deviation-error of the calibration given the reference points.
5. Compute for matrix-condition number.

"""

import sys
import os, sys
import numpy as np

import br2_vision
from br2_vision.dlt import DLT
from br2_vision.utility.logging import config_logging, get_script_logger


@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose")
def calibrate(verbose):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    # Configuration

    reference_point_filenames = config["PATHS"]["calibration_ref_points_path"]
    logger.debug(reference_point_filenames)

    # Read Configuration Points
    data = np.load(reference_point_filenames, allow_pickle=True)

    # Print datapoints
    for k, v in data.items():
        logger.debug(f"{k=}")
        logger.debug(f"{v.shape=}")
        logger.debug(f"{v=}")

    # DLT calibration
    dlt = DLT(calibration_path=config["PATHS"]["calibration_path"])
    for camera_id, points in data.items():
        camera_id = int(camera_id)
        dlt.add_camera(
            camera_id, calibration_type=11
        )  # TODO: enable more calibration types
        for point in points:  # point : (u,v,x,y,z)
            dlt.add_reference(*point, camera_id=camera_id)
    dlt.finalize(verbose=verbose)
    dlt.save()


if __name__ == "__main__":
    calibrate()
