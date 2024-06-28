import argparse
import os
import sys

import cv2
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from dlt import DLT
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

RUNID = 18
OVERLAY_CAM_ID = 3
PATH = "data_070521"

# DLT Load
CALIBRATION_PATH = os.path.join(PATH, "calibration")
dlt = DLT(calibration_path=CALIBRATION_PATH)
dlt.load()

# Read 3D coordinates
data = np.load(os.path.join(PATH, "postprocess", f"run-{RUNID}-position.npz"))
print(list(data.keys()))
cs_center = data["cross_section_center_position"]
# cs_center = data['position'].transpose([1,2,0])
cs_director = data["cross_section_director"]
print(cs_center.shape)
print(cs_director.shape)

# Overlay
cap = cv2.VideoCapture(
    os.path.join(PATH, "postprocess", f"cam-{OVERLAY_CAM_ID}-run-{RUNID}.mp4")
)
_, ori_frame = cap.read()
overlay_mask = np.zeros_like(ori_frame)

num_frame = 0
capture_frame_number = data["time"].shape[0] - 5
color = plt.get_cmap("hsv")(np.linspace(0, 1, 7)) * 255
while cap.isOpened():
    num_frame += 1
    ret, frame = cap.read()
    if not ret:
        break

    if num_frame == capture_frame_number:
        # Capture
        img = cv2.add(frame, overlay_mask)

        """
        # director mask
        director = cs_director[num_frame] 
        thickness = 5
        for i in range(5):
            for j in range(3):
                arrow = director[j,:,i]
                start_point = cs_center[num_frame,:,i]
                end_point = start_point + arrow * 0.01
                start_point = dlt.inverse_map(*start_point)[OVERLAY_CAM_ID]
                start_point = (int(start_point[0]), int(start_point[1]))
                end_point = dlt.inverse_map(*end_point)[OVERLAY_CAM_ID]
                end_point = (int(end_point[0]), int(end_point[1]))
                img = cv2.arrowedLine(img, start_point, end_point, color[i], thickness, tipLength=0.3)
        """

        cv2.imshow("result", img)
        cv2.waitKey(0)
        break

    # Draw the tracks
    for i in range(5):
        new = dlt.inverse_map(*cs_center[num_frame, :, i])
        old = dlt.inverse_map(*cs_center[num_frame, :, i])
        a, b = new[OVERLAY_CAM_ID]
        a = int(a)
        b = int(b)
        c, d = old[OVERLAY_CAM_ID]
        c = int(c)
        d = int(d)
        overlay_mask = cv2.line(overlay_mask, (a, b), (c, d), color[i].tolist(), 4)
        # frame = cv2.circle(frame,(a,b),13,color[i].tolist(),-1)

    """
    # display axis
    DISPLAY_AXIS = True
    if DISPLAY_AXIS:
        axis_length = 0.100
        # Frame 1
        uv = dlt_inverse_map(0,0,0, L, R)
        u = int(uv[0])
        v = int(uv[1])
        frame1 = cv2.circle(frame1, (u,v), 10, (0,0,0), -1)
        uv = dlt_inverse_map(*(O_OFFSET), L, R)
        u = int(uv[0])
        v = int(uv[1])
        frame1 = cv2.circle(frame1, (u,v), 30, (200,200,200), -1)
        # x-axis
        for i in np.linspace(0,axis_length,100):
            point = np.array([i,0,0])
            uv = dlt_inverse_map(*point, L, R)
            u = int(uv[0])
            v = int(uv[1])
            frame1 = cv2.circle(frame1, (u,v), 4, (255,0,0), -1)
        # y-axis
        for i in np.linspace(0,axis_length,100):
            point = np.array([0,i,0])
            uv = dlt_inverse_map(*point, L, R)
            u = int(uv[0])
            v = int(uv[1])
            frame1 = cv2.circle(frame1, (u,v), 4, (0,255,0), -1)
        # z-axis
        for i in np.linspace(0,axis_length,100):
            point = np.array([0,0,i])
            uv = dlt_inverse_map(*point, L, R)
            u = int(uv[0])
            v = int(uv[1])
            frame1 = cv2.circle(frame1, (u,v), 4, (0,0,255), -1)
        # Frame 2
        uv = dlt_inverse_map(0,0,0, L, R)
        u = int(uv[2])
        v = int(uv[3])
        frame2 = cv2.circle(frame2, (u,v), 10, (0,0,0), -1)
        uv = dlt_inverse_map(*(O_OFFSET), L, R)
        u = int(uv[2])
        v = int(uv[3])
        frame2 = cv2.circle(frame2, (u,v), 30, (200,100,100), -1)
        # x-axis
        for i in np.linspace(0,axis_length,100):
            point = np.array([i,0,0])
            uv = dlt_inverse_map(*point, L, R)
            u = int(uv[2])
            v = int(uv[3])
            frame2 = cv2.circle(frame2, (u,v), 4, (255,0,0), -1)
        # y-axis
        for i in np.linspace(0,axis_length,100):
            point = np.array([0,i,0])
            uv = dlt_inverse_map(*point, L, R)
            u = int(uv[2])
            v = int(uv[3])
            frame2 = cv2.circle(frame2, (u,v), 4, (0,255,0), -1)
        # z-axis
        for i in np.linspace(0,axis_length,100):
            point = np.array([0,0,i])
            uv = dlt_inverse_map(*point, L, R)
            u = int(uv[2])
            v = int(uv[3])
            frame2 = cv2.circle(frame2, (u,v), 4, (0,0,255), -1)
    """

    # Rescale frame
    # scale = 0.5
    # width = int(frame.shape[1]*scale)
    # height = int(frame.shape[0]*scale)
    # disp_frame = cv2.resize(frame, (width, height))

cap.release()
cv2.destroyAllWindows()
