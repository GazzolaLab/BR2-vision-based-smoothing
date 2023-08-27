import cv2
import numpy as np

def scale_image(frame, scale=1.0, interpolation=cv2.INTER_AREA):
    # Extract Control Frame
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (width, height), interpolation=interpolation)
    return frame
