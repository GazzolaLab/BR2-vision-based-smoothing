import math
from dataclasses import dataclass

import click
import cv2
import numpy as np


@dataclass
class Line:
    x1: int | None = None
    y1: int | None = None
    x2: int | None = None
    y2: int | None = None

    def draw(self, frame):
        # Green color in BGR
        if self.x1 is None or self.x2 is None or self.y1 is None or self.y2 is None:
            return
        color = (0, 235, 0)
        cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), color, 2)

    def clear(self):
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None


def mouse_callback(event, x, y, flags, param):
    line_start = param["line_start"]
    line_end = param["line_end"]

    if event == cv2.EVENT_LBUTTONDOWN:
        if line_start.x1 is None:
            line_start.x1, line_start.y1 = x, y
        elif line_start.x2 is None:
            line_start.x2, line_start.y2 = x, y
        elif line_end.x1 is None:
            line_end.x1, line_end.y1 = x, y
        elif line_end.x2 is None:
            line_end.x2, line_end.y2 = x, y


@click.command()
@click.option(
    "-p",
    "--path",
    type=click.Path(exists=True),
    help="Path to the video file.",
)
def main(path):
    line_start = Line()
    line_end = Line()

    cap = cv2.VideoCapture(path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    while True:
        # Read the video at the frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        line_start.draw(frame)
        line_end.draw(frame)
        cv2.imshow("Frame", frame)

        # Set the mouse callback
        param = {"line_start": line_start, "line_end": line_end}
        cv2.setMouseCallback("Frame", mouse_callback, param)

        # Wait for the user to press 'x' to quit
        # For q: -5 frame, w: -1 frame, e: +1 frame, r: +5 frame
        key = cv2.waitKey(0)
        if key == ord("q"):
            frame_idx = max(0, frame_idx - 5)
        elif key == ord("w"):
            frame_idx = max(0, frame_idx - 1)
        elif key == ord("e"):
            frame_idx = min(total_frame - 1, frame_idx + 1)
        elif key == ord("r"):
            frame_idx = min(total_frame - 1, frame_idx + 5)
        elif key == ord("x"):
            break
        elif key == ord("c"):
            line_start.clear()
            line_end.clear()
        elif key == ord("v"):
            # find cosine of the angle between two lines
            dx1 = line_start.x2 - line_start.x1
            dy1 = line_start.y2 - line_start.y1
            dx2 = line_end.x2 - line_end.x1
            dy2 = line_end.y2 - line_end.y1
            dot = dx1 * dx2 + dy1 * dy2
            det = dx1 * dy2 - dy1 * dx2
            angle_rad = math.atan2(det, dot)
            print(f"Angle: {angle_rad} radiian")

    cap.release()
    cv2.destroyAllWindows()
