import cv2
import numpy as np


def scale_image(frame, scale=1.0, interpolation=cv2.INTER_AREA):
    # Extract Control Frame
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (width, height), interpolation=interpolation)
    return frame


def flat_color(frame):
    """
    Convert color image to flat image (gray)

    Separater function is implemented in case of future color processing
    """
    flat = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return flat
