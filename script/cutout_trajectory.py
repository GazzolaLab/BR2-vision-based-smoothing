"""
Inquiry-based optical flow
"""
import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QInputDialog

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--camid", type=int, default=1)
parser.add_argument("--runid", type=int, default=1)
parser.add_argument("--reverse", action="store_true")
args = parser.parse_args()

CAMID = args.camid
RUNID = args.runid

path = "./data_090221/postprocess"
trajectory_path = os.path.join(path, f"cam-{CAMID}-footage-{RUNID}-ps.npz")

# Select points
assert os.path.exists(trajectory_path), "Initial reference points are not selected."
data = np.load(trajectory_path, allow_pickle=True)
inquiries = data["inquiry"]
all_tags = data["tags"].tolist()
all_points = data["points"]
history = list(data.get("history", []))

print(all_tags)
tag = input("tag name : ")
idx = all_tags.index(tag)
t = input("end-time for tag {}: ".format(tag))
if t != "" and int(t) > 0:
    if args.reverse:
        all_points[: int(t), idx, ...] = 0
    else:
        all_points[int(t) :, idx, ...] = 0

np.savez(
    trajectory_path,
    points=all_points,
    tags=all_tags,
    inquiry=inquiries,
    history=history,
)
