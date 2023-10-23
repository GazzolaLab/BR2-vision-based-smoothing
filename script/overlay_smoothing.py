import os
import sys
import numpy as np
import cv2
import pickle as pkl

from dlt import DLT
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.cm as cm

from scipy import interpolate

from config import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--runid", type=int, default=1)
args = parser.parse_args()

RUNID = args.runid

vector_length = 0.02
# Bring matplotlib color order into BGR
color_scheme = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def hex_to_bgr(hex):
    hex = hex.lstrip("#")
    hlen = len(hex)
    ret = [int(hex[i : i + hlen // 3], 16) for i in range(0, hlen, hlen // 3)]
    ret = tuple([ret[2], ret[1], ret[0]])
    return ret


color_scheme = [hex_to_bgr(color) for color in color_scheme]

# Configuration : Path
PATH = "data_090221"
SMOOTHING_DIR = "smoothing"
SMOOTHED_FILE_PATH = os.path.join(
    PATH, SMOOTHING_DIR, "smoothing-run-{}.pickle".format(RUNID)
)

# MISC.
# CENTER_OFFSET = np.array([0.15746584, 0.04388805, 0.24837398]) # base for 082021 serial BR2
# CENTER_OFFSET = np.array([0.19691901, 0.09749848, 0.26371042]) # base for 071221 single BR2 ( not necessary )
CENTER_OFFSET = np.array(
    [0.06718874, 0.08053348, 0.23180589]
)  # base for 090221 single BR2

# Read smoothed 3D coordinates
with open(SMOOTHED_FILE_PATH, "rb") as f:
    smoothed_data = pkl.load(f)
    smoothed_file_name = smoothed_data["datafile_name"]
    smoothed_time = np.array(smoothed_data["time"])[1:]
    smoothed_position = np.array(smoothed_data["position"])[1:]
    smoothed_director = np.array(smoothed_data["director"])[1:]
print(list(smoothed_data.keys()))
print(f"{smoothed_file_name=}")

# Read Experiment Data
exp_data = np.load(PREPROCESSED_POSITION_PATH.format(RUNID), allow_pickle=True)
exp_time = np.array(exp_data["time"])
exp_time = exp_time[exp_time < smoothed_time.max()]

# Interpolation (Match smoothed data into experiment video frames)
interp_position = interpolate.interp1d(smoothed_time, smoothed_position, axis=0)(
    exp_time
)
interp_director = interpolate.interp1d(smoothed_time, smoothed_director, axis=0)(
    exp_time
)
# interp_position[:,[0,2],:] = -interp_position[:,[0,2],:] # rotate (?)

# Interpolation Downsample
# interp_position = 0.5*(interp_position[...,1:] + interp_position[...,:-1])
interp_director = np.pad(
    interp_director, ((0, 0), (0, 0), (0, 0), (1, 0)), mode="linear_ramp"
)

# Offset
interp_position += CENTER_OFFSET[None, :, None]

# DLT Load
dlt = DLT(calibration_path=CALIBRATION_PATH)
dlt.load()
uvs_position = []  # list[time]:dict[camera id]:tuple(u,v)
uvs_director_d1 = []
uvs_director_d2 = []
uvs_director_d3 = []
for position in interp_position:
    uvs_position.append(dlt.inverse_map(*position))
for position, director in zip(interp_position, interp_director):
    uvs_director_d1.append(
        dlt.inverse_map(*(position + vector_length * director[0, :, :]))
    )
    uvs_director_d2.append(
        dlt.inverse_map(*(position + vector_length * director[1, :, :]))
    )
    uvs_director_d3.append(
        dlt.inverse_map(*(position + vector_length * director[2, :, :]))
    )
print(f"{len(uvs_position)=}")

# Overlay for each camera
num_camera = 5
for camid in tqdm(range(1, num_camera + 1), position=0):
    cap = cv2.VideoCapture(PREPROCESSED_FOOTAGE_VIDEO_PATH.format(camid, RUNID))
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f'{camid=} : {video_length=}')

    ret, frame = cap.read()
    frame_width = int(cap.get(4))
    frame_height = int(cap.get(3))
    video_size = (frame_height, frame_width)
    # print(f'{video_size=}')
    # print(f'{frame.shape=}')
    writer = cv2.VideoWriter(
        OVERLAY_SMOOTHING_VIDEO_PATH.format(camid, RUNID),
        cv2.VideoWriter_fourcc(*"mp4v"),
        60,
        video_size,
    )

    num_frame = 0
    pbar = tqdm(total=video_length, position=1)
    while cap.isOpened() and ret:
        if num_frame >= len(uvs_position):
            break
        pbar.update(1)

        # Create each layer
        layer_centerline = frame.copy()
        layer_d1fill = frame.copy()
        layer_d2fill = frame.copy()
        layer_d3fill = frame.copy()
        layer_d1lines = np.zeros_like(frame)
        layer_d2lines = np.zeros_like(frame)
        layer_d3lines = np.zeros_like(frame)

        color_centerline = (230, 230, 230)
        color_d1 = (50, 230, 50)  # color_scheme[0] #(50,230,50)
        color_d2 = (230, 50, 50)  # color_scheme[1] #(230,50,50)
        color_d3 = (50, 50, 230)  # color_scheme[2] #(50,50,230)

        contour_d1 = []
        contour_d2 = []
        contour_d3 = []

        # Draw the Elements
        skip_every = 6  # director line
        idx = -1
        for u, v, d1u, d1v, d2u, d2v, d3u, d3v in zip(
            *uvs_position[num_frame][camid],
            *uvs_director_d1[num_frame][camid],
            *uvs_director_d2[num_frame][camid],
            *uvs_director_d3[num_frame][camid],
        ):
            contour_d1.append((d1u, d1v))
            contour_d2.append((d2u, d2v))
            contour_d3.append((d3u, d3v))
            contour_d1.insert(0, (u, v))
            contour_d2.insert(0, (u, v))
            contour_d3.insert(0, (u, v))
            idx += 1
            if idx % skip_every != 0:
                continue
            u = int(u)
            v = int(v)
            d1u = int(d1u)
            d1v = int(d1v)
            d2u = int(d2u)
            d2v = int(d2v)
            d3u = int(d3u)
            d3v = int(d3v)
            layer_centerline = cv2.circle(
                layer_centerline, (u, v), 9, color_centerline, -1
            )
            layer_d1lines = cv2.line(
                layer_d1lines, (u, v), (d1u, d1v), color_d1, 4
            )  # d1
            layer_d2lines = cv2.line(
                layer_d2lines, (u, v), (d2u, d2v), color_d2, 4
            )  # d2
            layer_d3lines = cv2.line(
                layer_d3lines, (u, v), (d3u, d3v), color_d3, 4
            )  # d3

        contour_d1 = np.array([contour_d1], dtype=np.int32)
        contour_d2 = np.array([contour_d2], dtype=np.int32)
        contour_d3 = np.array([contour_d3], dtype=np.int32)
        cv2.fillPoly(layer_d1fill, pts=contour_d1, color=color_d1)
        cv2.fillPoly(layer_d2fill, pts=contour_d2, color=color_d2)
        cv2.fillPoly(layer_d3fill, pts=contour_d3, color=color_d3)

        layers = [frame, layer_centerline, layer_d1fill, layer_d2fill, layer_d3fill]
        layers_weights1 = [1.0, 1.0, 1.0, 1.0]
        layers_weights2 = [0.0, 0.3, 0.3, 0.0]
        # layers_weights = np.array(layers_weights) / np.sum(layers_weights)

        disp_img = layers[0]
        for layer, w1, w2 in zip(layers[1:], layers_weights1, layers_weights2):
            w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)
            disp_img = cv2.addWeighted(disp_img, w1, layer, w2, 0)

        disp_img = cv2.add(disp_img, layer_d1lines)
        disp_img = cv2.add(disp_img, layer_d2lines)
        # disp_img = cv2.add(disp_img, layer_d3lines)

        disp_img = cv2.polylines(disp_img, contour_d1, True, (255, 255, 255), 1)
        disp_img = cv2.polylines(disp_img, contour_d2, True, (255, 255, 255), 1)
        # disp_img = cv2.polylines(disp_img, contour_d3, True, (255,255,255), 1)

        # Imshow (Debug)
        # cv2.imshow('disp_img', disp_img)
        # cv2.waitKey(0)

        # Write
        writer.write(disp_img)

        # Iterate update
        ret, frame = cap.read()
        num_frame += 1

    cap.release()
    writer.release()
cv2.destroyAllWindows()
