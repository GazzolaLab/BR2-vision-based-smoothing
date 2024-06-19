import os
import sys

import cv2
import numpy as np


def cv2_draw_cross_indicator(frame, x, y, l, color=(270, 20, 20)):
    """Draw Cross Indicator

    Parameters
    ----------
    frame : numpy array
        Image to draw
    x : int
        x-coordinate
    y : int
        y-coordinate
    l : int
        size of the indicator
    color : tuple(int,int,int)
        RGB color in 0-255 range
    """
    # Draw two triangle (top and bottom)
    ptc = (x, y)
    ptlb = (x - l, y - l)
    ptrb = (x + l, y - l)
    ptlt = (x - l, y + l)
    ptrt = (x + l, y + l)
    bottom_triangle = np.array([ptc, ptlb, ptrb])
    upper_triangle = np.array([ptc, ptlt, ptrt])
    cv2.drawContours(frame, [bottom_triangle], 0, color, -1)
    cv2.drawContours(frame, [upper_triangle], 0, color, -1)


def cv2_draw_label(
    frame,
    x,
    y,
    s,
    color=(270, 20, 20),
    length=5,
    fontScale=0.5,
    fontColor=(255, 255, 255),
):
    """Draw Label

    Drawing combination of label and cross-indicator at a point (x,y)

    Parameters
    ----------
    frame : numpy array
        Image to draw
    x : int
        x-coordinate
    y : int
        y-coordinate
    s : str
        text (castable)
    color : tuple(int,int,int)
        RGB color in 0-255 range
    length :
        length of the cross-indicator
    """

    # Plot circle
    cv2_draw_cross_indicator(frame, x, y, length, color)

    # Plot label
    cv2.putText(
        frame,
        str(s),
        org=(x + 5, y + 15),  # bottom Left Corner Of Text
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=fontColor,
        # thickness=7,
        lineType=2,
    )
