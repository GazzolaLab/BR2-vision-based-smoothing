import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from nptyping import NDArray
from scipy.interpolate import CubicSpline
from tqdm import tqdm

import br2_vision
from br2_vision.cv2_custom.extract_info import get_video_frame_count
from br2_vision.cv2_custom.transformation import flat_color, scale_image
from br2_vision.data_structure import FlowQueue, MarkerPositions, TrackingData
from br2_vision.utility.logging import config_logging, get_script_logger

# from collections import defaultdict


# from dlt import DLT
# from cv2_custom.marking import cv2_draw_label


# from sklearn.linear_model import LinearRegression
# from scipy.spatial.distance import directed_hausdorff
# from scipy.signal import savgol_filter as sgfilter


def mouse_click_event(event, x, y, flags, param):
    data = param["data"]
    current_frame_idx = param["current_frame_idx"][0]
    if event == cv2.EVENT_LBUTTONDOWN:
        data[current_frame_idx, 0] = x
        data[current_frame_idx, 1] = y

        # display
        param["func_display"](
            param["frames"][current_frame_idx].copy(),
            data,
            param["window_name"],
        )


# Optical Flow and Point Detection Module
class ManualTracing:
    """
    ManualTracing class is a module to track the points in the video manually.
    """

    # Configuration: Color scheme
    _color = np.random.randint(0, 235, (100, 3)).astype(int)  # 100 points for now

    def __init__(
        self,
        video_path,
        flow_queue: FlowQueue,
        dataset: TrackingData,
        scale: float = 1.5,
    ):
        self.video_path = video_path

        self.flow_queue = flow_queue
        self.dataset = dataset
        self.scale = scale

        self.__num_frames = None

    @property
    def num_frames(self):
        """
        Get the number of frames in the video
        """
        if self.__num_frames is None:
            self.__num_frames = get_video_frame_count(self.video_path)
        return self.__num_frames

    def run(self, debug=False) -> bool:
        """
        For all queues:
            Run optical flow from start_frame to end_frame
            If there are other queue with the same start_frame and end_frame, group them, run together (inquiry)
        debug mode: dry run
        """

        start_frame = self.flow_queue.start_frame
        end_frame = self.flow_queue.end_frame

        if debug:
            print(f"Dry Run: Start: {start_frame}, End: {end_frame}.")
            return

        # Run inquiry
        data = self.trace(start_frame, end_frame)
        if data is None:
            # Halting
            return False

        # Interpolation
        indices = np.where(data[:, 0] != -1)[0]
        if len(indices) < 2:
            raise ValueError("More point needs to be selected")
        for si, ei in zip(indices[:-1], indices[1:]):
            data[si:ei] = np.linspace(data[si], data[ei], ei-si)
        #points = data[indices, :]
        #spline = CubicSpline(indices, points)
        #data = spline(np.arange(end_frame - start_frame))

        # Save
        q = self.flow_queue
        self.dataset.save_pixel_flow_trajectory(data, q, self.num_frames)
        q.done = True

        return True

    def draw_points(
        self, frame, points, radius=2, color=(0, 235, 0), thickness=-1
    ):  # pragma: no cover
        # draw the points (overlay)
        for i, point in enumerate(points):
            a, b = point.ravel()
            a, b = int(a), int(b)
            frame[:] = cv2.circle(frame, (a, b), radius, color, thickness)

    def draw_track(self, frame, p0, p1, color=(0, 235, 0)):  # pragma: no cover
        # draw the tracks (overlay)
        for i, (new, old) in enumerate(zip(p1, p0)):
            a, b = new.ravel()
            a, b = int(a), int(b)
            c, d = old.ravel()
            c, d = int(c), int(d)
            frame[:] = cv2.line(frame, (a, b), (c, d), color, 2)

    def stack_frames(self, start_frame: int, end_frame: int):
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened(), "Video is not properly opened: {}".format(
            self.video_path
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Jump frame

        # get width and height
        width = int(cap.get(3) * self.scale)
        height = int(cap.get(4) * self.scale)

        # get frames
        frames = np.zeros((end_frame - start_frame, height, width, 3), np.uint8)
        for i in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frame = scale_image(frame, self.scale)
            frames[i] = frame

        cap.release()
        return frames

    def display(
        self,
        frame: NDArray,
        data_collection: NDArray,
        name: str,
    ):
        indices = np.where(data_collection[:, 0] != -1)[0]
        num_points = len(indices)
        if num_points == 0:
            pass
        elif num_points == 1:
            self.draw_points(frame, data_collection[indices, :])
        else:
            points = data_collection[indices, :]
            self.draw_points(frame, points)
            self.draw_track(frame, points[:-1], points[1:])

        cv2.imshow(name, frame)

    # It is excluded from coverage test: code is mostly based on cv2.
    def trace(self, start_frame, end_frame, debug=False):  # pragma: no cover
        # initialize data_collection: -1
        data_length = end_frame - start_frame
        data_collection = np.zeros((data_length, 2), dtype=np.int_) - 1

        frames = self.stack_frames(start_frame, end_frame)

        # Set initial points
        current_frame_idx = [0]
        point = self.flow_queue.point
        data_collection[0, :] = point

        window_name = "Tracing"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(
            window_name,
            mouse_click_event,
            param={
                "data": data_collection,
                "current_frame_idx": current_frame_idx,
                "frames": frames,
                "window_name": window_name,
                "func_display": self.display,
            },
        )

        print(
            "q: -5 frames, w: -1 frame, e: +1 frame, r: +5 frames, x: exit (without save)"
        )
        print("^: first frame, $: last frame")
        print("z: delete last trace point")
        prev_frame = -1
        force_refresh = False
        while current_frame_idx[0] < data_length:
            # Only redraw when the frame is changed
            if prev_frame != current_frame_idx[0] or force_refresh:
                prev_frame = current_frame_idx[0]
                self.display(
                    frames[current_frame_idx[0]].copy(),
                    data_collection,
                    window_name,
                )
                print(
                    f"(frame {current_frame_idx[0]+start_frame}){current_frame_idx[0]}/{data_length}"
                )
                force_refresh = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                current_frame_idx[0] = max(0, current_frame_idx[0] - 5)
            elif key == ord("w"):
                current_frame_idx[0] = max(0, current_frame_idx[0] - 1)
            elif key == ord("e"):
                current_frame_idx[0] = current_frame_idx[0] + 1
            elif key == ord("r"):
                current_frame_idx[0] = current_frame_idx[0] + 5
            elif key == ord("^"):
                current_frame_idx[0] = 0
            elif key == ord("$"):
                current_frame_idx[0] = data_length - 1
            elif key == ord("x"):
                data_collection = None
                break
            elif key == ord("z"):
                where = np.where(data_collection[:, 0] != -1)[0]
                if len(where) > 0:
                    data_collection[:, where[-1]] = -1
                force_refresh = True

        cv2.destroyAllWindows()
        return data_collection
