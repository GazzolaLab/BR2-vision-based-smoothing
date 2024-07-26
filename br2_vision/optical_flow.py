import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
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


# Optical Flow and Point Detection Module
class CameraOpticalFlow:
    """
    Optical Flow and Point Detection Module

    Given a video and list of FlowQueue, this module calculates the optical flow
    and save the trajectory of the points in the dataset.
    """

    # Configuration: Corner detection
    # parameters for ShiTomasi corner detection
    _feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Configuration: Corner SubPix
    _subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    # Configuration: Optical Flow
    # Parameters for lucas kanade optical flow
    _lk_params = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.0001),
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=0.000,
    )  # 0.017

    # Configuration: Color scheme
    _color = np.random.randint(0, 235, (100, 3)).astype(int)  # 100 points for now

    def __init__(
        self,
        video_path,
        flow_queues: list[FlowQueue],
        dataset: TrackingData,
        scale: float = 1.0,
        force_run_all: bool = False,
    ):
        self.video_path = video_path

        if force_run_all:
            self.flow_queues = flow_queues
        else:
            self.flow_queues = [q for q in flow_queues if not q.done]
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

    def run(self, debug=False):
        """
        For all queues:
            Run optical flow from start_frame to end_frame
            If there are other queue with the same start_frame and end_frame, group them, run together (inquiry)
        debug mode: dry run
        """

        # Group queues
        indices = [
            i for i in range(len(self.flow_queues)) if not self.flow_queues[i].done
        ]
        while indices:
            base_index = indices[0]
            inquiry = [base_index]
            start_frame = self.flow_queues[base_index].start_frame
            end_frame = self.flow_queues[base_index].end_frame

            # Collect all queues with same start_frame and end_frame
            for i in indices[1:]:
                q = self.flow_queues[i]
                if q.start_frame == start_frame and q.end_frame == end_frame:
                    inquiry.append(i)
            [indices.remove(i) for i in inquiry]

            if debug:
                print(
                    f"Dry Run: Start: {start_frame}, End: {end_frame}, Inquiry: {inquiry}"
                )
                for i in inquiry:
                    print("  ", self.flow_queues[i])
                continue

            # Run inquiry
            data_collection, _ = self.next_inquiry(inquiry, start_frame, end_frame)
            # TODO: collect errors and save as well

            # Save
            assert len(inquiry) == len(
                data_collection
            ), f"{len(inquiry)} != {len(data_collection)}"
            for i, data in zip(inquiry, data_collection):
                q = self.flow_queues[i]
                self.dataset.save_pixel_flow_trajectory(data, q, self.num_frames)
                q.done = True

    def get_points_in_order(self, keys):
        points = []
        for key in keys:
            points.append(self.p[key])
        return points

    def get_point_info(self, key):
        info = {}
        info["point"] = self.p[key]
        info["is_occluded"] = self.is_occluded[key]
        if self.is_occluded[key]:
            info["predicted_point"] = self.reappearance_module.get_predicted_point(key)
        return info

    def get_current_frame(self, mark_trajectory=False, mark_points=False):
        frame = self.current_frame.copy()
        if mark_trajectory:
            color = plt.get_cmap("hsv")(np.linspace(0, 1, 20)) * 255
            trajectories = self.reappearance_module.trajectory_block[mark_trajectory]
            for idx, trajectory in enumerate(trajectories):
                if len(trajectory) <= 1:
                    continue
                self.draw_track(frame, trajectory[:-1], trajectory[1:], color[idx])
                if mark_points:
                    for point in trajectory:
                        frame = cv2.circle(
                            frame, (int(point[0]), int(point[1])), 13, color[idx], -1
                        )
        return frame

    def get_none_occluded_points(self):
        points = []
        keys = []
        for k, is_occluded in self.is_occluded.items():
            if is_occluded:
                continue
            points.append(self.p[k])
            keys.append(k)
        return np.array(points), keys

    def get_none_occluded_points_dict(self):
        points = {}
        for k, is_occluded in self.is_occluded.items():
            if is_occluded:
                continue
            points[k] = self.p[k]
        return points

    def draw_points(
        self, frame, points, radius=8, color=(0, 235, 0), thickness=-1
    ):  # pragma: no cover
        # draw the points (overlay)
        for i, point in enumerate(points):
            a, b = point.ravel()
            a, b = int(a), int(b)
            frame[:] = cv2.circle(frame, (a, b), radius, color, thickness)

    def draw_track(self, mask, p0, p1, color=(0, 235, 0)):  # pragma: no cover
        # draw the tracks (overlay)
        for i, (new, old) in enumerate(zip(p1, p0)):
            a, b = new.ravel()
            a, b = int(a), int(b)
            c, d = old.ravel()
            c, d = int(c), int(d)
            # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            mask[:] = cv2.line(mask, (a, b), (c, d), color, 2)

    def show_frame(self, frame, mask):
        img = cv2.add(frame, mask)
        for key in USED_TAGS:
            a, b = self.p[key]
            a, b = int(a), int(b)
            if self.is_occluded[key]:  # Ocluded points are red x
                cv2.drawMarker(
                    img,
                    (a, b),
                    color=(0, 0, 235),
                    markerType=cv2.MARKER_CROSS,
                    thickness=2,
                )
            else:  # visible points are green o
                img = cv2.circle(img, (a, b), 8, (0, 235, 0), -1)
        cv2.imshow("frame", img)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # FIXME: Change it to be global functionals
    def render_tracking_video(self, save_path, camera_id):  # pragma: no cover
        """save_tracking_video

        Parameters
        ----------
        save_path :
            path
        """
        print(f"{save_path=}")
        print("Saving tracing video ...")
        import tempfile

        tmp_path = os.path.join(tempfile.gettempdir(), "_r.mp4")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        # Create a mask image for drawing purposes
        ret, old_frame = cap.read()
        old_frame = scale_image(old_frame, self.scale)
        mask = np.zeros_like(old_frame)

        frame_width = int(old_frame.shape[0])
        frame_height = int(old_frame.shape[1])
        writer = cv2.VideoWriter(
            tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 60, (frame_height, frame_width)
        )
        video_length = self.num_frames

        data_collection = []
        tags = []
        for tag, data in self.dataset.iter_trajectory(camera_id):
            data_collection.append(data)
            tags.append(tag)
        data_collection = np.asarray(data_collection)

        # data_collection = np.zeros((len(queues), video_length, 2), dtype=np.int_) - 1
        # for qid, q in enumerate(queues):
        #    _data = self.dataset.load_pixel_flow_trajectory(q, full_trajectory=True)
        #    if _data is not None:
        #        data_collection[qid, :, :] = _data

        for num_frame in tqdm(range(video_length), miniters=10):
            # while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = scale_image(frame, self.scale)

            # draw the tracks
            good_new = data_collection[:, num_frame + 1, :]
            good_old = data_collection[:, num_frame, :]
            for qid, (new, old) in enumerate(zip(good_new, good_old)):
                tag = tags[qid]  # queues[qid].get_tag()
                a, b = new.ravel()
                a, b = int(a), int(b)
                c, d = old.ravel()
                c, d = int(c), int(d)
                if (a <= 5 and b <= 5) or (c <= 5 and d <= 5):
                    continue
                mask = cv2.line(
                    mask, (a, b), (c, d), CameraOpticalFlow._color[qid].tolist(), 2
                )
                _frame = frame.copy()
                _frame = cv2.circle(
                    _frame, (a, b), 7, CameraOpticalFlow._color[qid].tolist(), -1
                )
                alpha = 0.6  # Transparency factor.
                # Following line overlays transparent circles over the image
                frame = cv2.addWeighted(_frame, alpha, frame, 1 - alpha, 0)

                text_img = np.zeros_like(frame)
                cv2.putText(
                    text_img,
                    tag,
                    (a, b + 25),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    lineType=2,
                )
                text_img = cv2.warpAffine(
                    text_img,
                    cv2.getRotationMatrix2D((a, b), 20, 1),  # rotate 30degree
                    (frame.shape[1], frame.shape[0]),
                )
                frame = cv2.add(frame, text_img)

            img = cv2.add(frame, mask)
            assert img.shape == (frame_width, frame_height, 3)
            writer.write(img)
        cap.release()
        # cv2.destroyAllWindows()
        writer.release()

        # TODO
        import subprocess

        command = ["ffmpeg", "-y"]
        command.extend(["-i", tmp_path])
        command.extend(["-vf", "\"drawtext=text=\'Frame %{n}\':fontsize=30:fontcolor=white:borderw=2:bordercolor=black:x=10:y=th+10\""])
        command.extend(["-c:a", "copy"])
        command.extend([save_path])
        command = " ".join(command)
        print("running : ", command)
        sts = subprocess.Popen(command, shell=True).wait()

    # It is excluded from coverage test: code is mostly based on cv2.
    def next_inquiry(self, inquiry, stime, etime, debug=False):  # pragma: no cover
        num_queue = len(inquiry)
        # initialize data_collection: -1
        data_length = etime - stime
        data_collection = np.zeros((num_queue, data_length, 2), dtype=np.int_) - 1

        # Forward Flow
        # Load video
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened(), "Video is not properly opened: {}".format(
            self.video_path
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, stime)  # Jump frame

        # Read first frames
        ret, frame = cap.read()
        frame = scale_image(frame, self.scale)
        old_gray = flat_color(frame)

        # Set initial points
        for idx, qi in enumerate(inquiry):
            point = self.flow_queues[qi].point  # (x, y)
            data_collection[idx, 0, :] = point
        p0 = data_collection[:, 0, :].reshape(-1, 1, 2).astype(np.float32)

        errors = []
        status = None
        for num_frame in tqdm(range(data_length - 1)):
            ret, frame = cap.read()
            if not ret:
                break
            frame = scale_image(frame, self.scale)
            frame_gray = flat_color(frame)
            # Preprocess (sharpen)
            # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            # frame_gray = cv2.filter2D(frame_gray, -1, sharpen_kernel)

            # Calculate optical flow
            p1, new_status, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **CameraOpticalFlow._lk_params
            )  # p1: (n, 1, 2), new_status: (n, 1), err: (n, 1)
            if status is None:
                status = new_status[:, 0].astype(np.bool_)
            else:
                status = np.logical_and(status, new_status[:, 0])
            if np.all(~status):
                # If all points are lost, stop the flow
                break

            # Record
            err[~status] = np.nan
            errors.append(err)
            data_collection[status, num_frame + 1, :] = p1[status, 0, :].astype(np.int_)

            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = p1.reshape(-1, 1, 2)

        cap.release()
        cv2.destroyAllWindows()
        return data_collection, errors
