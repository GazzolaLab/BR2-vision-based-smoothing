import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from cv2_custom.marking import cv2_draw_label
from cv2_custom.transformation import scale_image

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QInputDialog

from config import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--camid", type=int, default=1)
parser.add_argument("--runid", type=int, default=1)
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument("--end_frame", type=int, default=-1)
args = parser.parse_args()

CAMID = args.camid
RUNID = args.runid

# BR2 Configuration
NUM_RING = 5
NUM_POINT = 9
RING_CHAR = "R"

# Path
path = POSTPROCESSING_PATH
video_path = PREPROCESSED_FOOTAGE_VIDEO_PATH.format(CAMID, RUNID)
video_name = os.path.basename(video_path)
initial_point_file = TRACKING_FILE.format(
    CAMID, RUNID
)  # Initial point shared by experiment (save path)

# Set Colors
np.random.seed(100)
color = np.random.randint(0, 235, (100, 3)).astype(int)

# Capture Video
cap = cv2.VideoCapture(os.path.join(path, video_name))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.start_frame == -1:
    setattr(args, "start_frame", length - 1)
    # args.start_frame = length-1
cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
ret, curr_frame = cap.read()
if args.end_frame == -1:
    args.end_frame = length

assert args.start_frame < length

app = QApplication(sys.argv)
tags = []
points = []

# Mouse Handle
prev_tag = ""


def mouse_event_click_point(event, x, y, flags, param):
    global prev_tag
    points = param["points"]
    tags = param["tags"]
    bypass_inquiry = flags & cv2.EVENT_FLAG_CTRLKEY
    if event == cv2.EVENT_LBUTTONDOWN:
        point = np.array([x, y], dtype=np.int32).reshape([1, 2])
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Second zoom-layer selection
        uv = zoomed_inquiry(param["frame"], np.array([x, y]))
        point = uv.astype(np.int32).reshape([1, 2])
    else:
        return
    points.append(point)

    # Ask for a tag in a separate window
    if bypass_inquiry:
        tag = prev_tag
    else:
        _ok = False
        while not _ok:
            tag, _ok = QInputDialog.getText(
                QWidget(), "Tag", "Input Tag", text=prev_tag
            )
    if tag[0] == "R":
        prev_tag = (
            tag[0]
            + str(int(tag.split("-")[0][1:]) + 1)
            + "-"
            + str(int(tag.split("-")[1]))
        )
    else:
        prev_tag = tag
    tags.append(tag)
    print("added: ")
    print(point, tag)


def zoomed_inquiry(current_frame, uv, scale=5.0, disp_h=80, disp_w=80):
    x, y = uv
    x = int(x)
    y = int(y)

    # Region of interest display
    window_name_roi = "roi"
    cv2.namedWindow(window_name_roi)
    disp_img_roi = current_frame.copy()
    disp_img_roi = cv2.rectangle(
        disp_img_roi,
        (x - disp_w // 2, y - disp_h // 2),
        (x + disp_w // 2, y + disp_h // 2),
        (0, 0, 255),
        thickness=3,
    )
    cv2.imshow(window_name_roi, disp_img_roi)

    # Transformation
    img = current_frame.copy()
    padded_img = cv2.copyMakeBorder(
        img,
        disp_h // 2,
        disp_h // 2,
        disp_w // 2,
        disp_w // 2,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    scaled_img = scale_image(padded_img[y : y + disp_h, x : x + disp_w], scale)
    _x = int(disp_w * scale / 2)
    _y = int(disp_h * scale / 2)
    _uv = np.array([_x, _y])

    # Implement mouse event for clicking other point
    original_uv = _uv.copy()

    def onMouse(event, x, y, flags, param):
        uv = param["uv"]
        original_uv = param["original_uv"]
        if event == cv2.EVENT_LBUTTONDOWN:
            uv[0] = x
            uv[1] = y
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Return original uv
            uv[:] = original_uv

    # Inquiry Loop
    inquiry_on = True
    window_name = "select reappeared point"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name, onMouse, param={"uv": _uv, "original_uv": original_uv}
    )
    while inquiry_on:
        disp_img = scaled_img.copy()

        # Draw cross with exact center _uv
        disp_img[_uv[1] : _uv[1] + 1, :] = np.array([0, 0, 235])
        disp_img[:, _uv[0] : _uv[0] + 1] = np.array([0, 0, 235])

        cv2.imshow(window_name, disp_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):  # Cancel: No point found
            inquiry_on = False
            uv = original_uv
        elif key == ord("a"):  # Accept: accept change
            inquiry_on = False
        else:
            pass

    cv2.destroyWindow(window_name)
    cv2.destroyWindow(window_name_roi)

    x = int(_uv[0] / scale) + x - disp_w // 2
    y = int(_uv[1] / scale) + y - disp_h // 2

    return np.array([x, y], dtype=int)


# Draw
def frame_label(frame, points, tags):
    for inx in range(len(points)):
        point = tuple(points[inx][0])
        tag = tags[inx]
        cv2_draw_label(frame, int(point[0]), int(point[1]), tag, fontScale=0.8)


# First-layer Selection
cv2.namedWindow(video_name)
cv2.setMouseCallback(
    video_name,
    mouse_event_click_point,
    param={"frame": curr_frame, "points": points, "tags": tags},
)
while True:
    disp_img = curr_frame.copy()

    if len(points) > 0:
        frame_label(disp_img, points, tags)

    cv2.imshow(video_name, disp_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        print("done")
        break
    elif key == ord("d"):
        if len(points) > 0:
            points.pop(-1)
            tags.pop(-1)
            print("deleted")
    elif key == ord("p"):
        print("check")
        print(points)
        print(tags)
cv2.destroyAllWindows()

# Load existing points and tags
if os.path.exists(initial_point_file):
    data = np.load(initial_point_file, allow_pickle=True)
    all_tags = data["tags"].tolist()
    all_points = data["points"]
    inquiry = data["inquiry"].tolist()  # list((list(tag), stime, etime))
else:
    all_tags = []
    for ring_id in range(1, NUM_RING + 1):
        for point_id in range(NUM_POINT):
            all_tags.append(f"{RING_CHAR}{ring_id}-{point_id}")
    all_points = np.zeros([length, len(all_tags), 2], dtype=np.float32)
    inquiry = []

# Add Flow Queue
sframe = args.start_frame
eframe = args.end_frame
for tag, point in zip(tags, points):
    idx = all_tags.index(tag)
    all_points[sframe, idx, :] = point
inquiry.append((tags, sframe, eframe))

# Save points
np.savez(
    initial_point_file, points=all_points, tags=all_tags, inquiry=inquiry, history=[]
)
