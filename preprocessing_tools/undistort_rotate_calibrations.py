import cv2
import glob
import os, sys
import argparse

from undistort_tools.undistort import undistort

from config import *

"""
Undistort and rotate calibration frames.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--camid', type=int, default=1, help='Camera Index')
parser.add_argument('--rotate', type=str, default=None, help='Rotation in cv2 (ex:ROTATE_90_CLOCKWISE)')
args = parser.parse_args()

camid = args.camid
if args.rotate is not None:
    cv2_rotation = getattr(cv2, args.rotate, None)
else:
    cv2_rotation = None

image_path_collection = CALIBRATION_IMAGE_COLLECTION.format(camid)
raw_images = glob.glob(image_path_collection, recursive=True)

for path in raw_images:
    frame = cv2.imread(path)

    # Undistort
    frame = undistort(frame)

    # Rotate
    if cv2_rotation != None:
        frame = cv2.rotate(frame, cv2_rotation)
      
    # Write 
    cv2.imwrite(path, frame)
      
    print("The image was successfully saved - {}".format(path))
