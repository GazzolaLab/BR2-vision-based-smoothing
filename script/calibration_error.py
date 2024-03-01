import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

from dlt import DLT

from collections import defaultdict

from calibration_point_selection import label_to_3Dcoord

from config import *

PATH = "data_082421"
IMAGE_PATH = os.path.join(PATH, "calibration")
CALIBRATION_PATH = os.path.join(PATH, "calibration")

CAMERA_INDICATOR = "cam"
PURPOSE_INDICATOR = "calibration"

calibration_images = glob.glob(
    os.path.join(
        IMAGE_PATH, "{}-*-{}-*[0-9].png".format(CAMERA_INDICATOR, PURPOSE_INDICATOR)
    )
)

reference_image_paths = {}  # key: (Camera ID, x-location ID)
for path in calibration_images:
    base = os.path.basename(path)
    base = base.split(".")[0].split("-")
    reference_image_paths[(base[1], base[3])] = path

scale = 1.0

point_collection = defaultdict(dict)
for (camera_id, x_id), path in reference_image_paths.items():
    save_path_points = os.path.join(
        IMAGE_PATH, "cam-{}-calibration-{}-refpoints.npz".format(camera_id, x_id)
    )
    points = np.load(save_path_points)["coords"]
    # Save the points
    for u, v, y_id, z_id, _ in points:
        u = int(u / scale)
        v = int(v / scale)
        point_collection[(int(x_id), y_id, z_id)][int(camera_id)] = (u, v)

# Reconsturction loss
dlt = DLT(calibration_path=CALIBRATION_PATH)
dlt.load()
e_x = []
e_y = []
e_z = []
error_vector = []
condition_numbers = []
for k, uvs in point_collection.items():
    if len(uvs.keys()) <= 1:  # Requires least 2 camera observing the camera
        continue
    (dlt_x, dlt_y, dlt_z), cond = dlt.map(uvs)
    true_x, true_y, true_z = label_to_3Dcoord(*k)
    # Loss Definition
    e_x.append(dlt_x - true_x)
    e_y.append(dlt_y - true_y)
    e_z.append(dlt_z - true_z)
    error_vector.append(
        [true_x, true_y, true_z, dlt_x - true_x, dlt_y - true_y, dlt_z - true_z]
    )
    condition_numbers.append(cond)
e_x = np.array(e_x)
e_y = np.array(e_y)
e_z = np.array(e_z)
error_vector = np.array(error_vector)

er = np.sqrt(e_z ** 2 + e_y ** 2 + e_z ** 2)
print("Reconstruction Error (m): ", er.mean())
print(sps.describe(er))
print("Reconstruction Condition: ", np.mean(condition_numbers))
print(sps.describe(condition_numbers))
plt.figure("error (er) histogram")
plt.hist(er, 100)
plt.title("reconstruction error")
plt.figure("error bar in domain", figsize=(10, 8))
ax = plt.axes(projection="3d")
ax.set_title("reconstruction difference in domain")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
for vec in error_vector:
    ax.quiver(*vec)

plt.show()
