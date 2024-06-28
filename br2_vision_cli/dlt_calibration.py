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

import glob
import os
import re
import sys
from collections import defaultdict

import click
import numpy as np

import br2_vision
from br2_vision.dlt import DLT, label_to_3Dcoord
from br2_vision.utility.logging import config_logging, get_script_logger


@click.command()
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose")
def calibrate(verbose):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    # Collect reference points
    calibration_ref_point_save = config["PATHS"][
        "calibration_ref_point_save"
    ]  # (camera-id, x-id)

    # extract camera-id and x-id
    data = defaultdict(list)
    all_saves = glob.glob(calibration_ref_point_save.format("*", "*"))
    logger.debug(all_saves)
    for save in all_saves:
        expression = calibration_ref_point_save.format(r"(\d+)", r"(\d+)")
        camera_id, x_id = re.findall(expression, save)[0]
        camera_id = int(camera_id)
        x_id = int(x_id)

        save_path_points = calibration_ref_point_save.format(camera_id, x_id)
        points = np.load(save_path_points)["coords"]

        for u, v, y_id, z_id, _ in points:
            x, y, z = label_to_3Dcoord(x_id, y_id, z_id, config)
            u = int(u)
            v = int(v)
            data[str(camera_id)].append((u, v, x, y, z))

    output_name = config["PATHS"]["calibration_ref_points_path"]
    np.savez(output_name, **data)  # For debugging purpose.

    # Print datapoints
    for k, v in data.items():
        logger.debug(f"{k=}")
        logger.debug(f"{np.array(v).shape=}")

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
