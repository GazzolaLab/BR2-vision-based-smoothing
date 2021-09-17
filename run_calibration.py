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
import glob
import numpy as np
import scipy.stats as ss
import scipy.linalg as spl
import matplotlib.pyplot as plt

from dlt import DLT

from config import *

# Configuration
reference_point_filenames = CALIBRATION_REF_POINTS_PATH
print(reference_point_filenames)

# Read Configuration Points
data = np.load(reference_point_filenames, allow_pickle=True)

# Print datapoints 
for k, v in data.items():
    print(k, v.shape)
    print(v)

# DLT calibration
dlt = DLT(
    calibration_path=CALIBRATION_PATH
)
for camera_id, points in data.items():
    camera_id = int(camera_id)
    dlt.add_camera(camera_id, calibration_type=11)
    for point in points: # point : (u,v,x,y,z)
        dlt.add_reference(*point, camera_id=camera_id)
dlt.finalize(verbose=True)
dlt.save()

