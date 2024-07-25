from typing import TypeAlias
import operator
import os
from dataclasses import dataclass
from typing import List, Tuple

import h5py
import numpy as np
from nptyping import Floating, NDArray, Shape
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from br2_vision.utility.convert_coordinate import get_center_and_normal
from br2_vision.utility.logging import get_script_logger

from .tracking_data import TrackingData
from .utils import raise_if_outside_context

CameraID = TypeAlias("CameraID", int)


class OriginData:
    """
    Data structure for postures
    """

    def __init__(self, path):
        self.path = path
        self.logger = get_script_logger(os.path.basename(__file__))

        self._inside_context = False

        self._positions: NDArray[Shape["[x,y,z]"], Floating] | None = None
        self._positions_key: str = "/origin/xyz"

        self._origin_frames: NDArray[Shape["C, [u,v]"], Floating] | None = None
        self._origin_frames_key: str = "/origin/uv"

        self._camera_indices: list[CameraID] | None = None
        self._camera_indices_key: str = "/origin/camera_indices"

    @raise_if_outside_context
    def get_origin(self):
        return self._positions

    @raise_if_outside_context
    def get_camera_origin(self) -> dict[CameraID, NDArray[Shape["[u,v]"], Floating]]:
        return {
            for camera_id in self._camera_indices
        }

    @raise_if_outside_context
    def set_camera_frames(self, camera_ids: list[CameraID], frames: NDArray[Shape["C, [u,v]"], Floating]):
        self._camera_indices = camera_ids
        self._origin_frames = frames

    @raise_if_outside_context
    def set_origin_xyz(self, xyz: NDArray[Shape["[x,y,z]"], Floating]):
        self._positions = xyz

    @raise_if_outside_context
    def save(self):
        with h5py.File(self.path, "a") as h5f:
            if self._positions is not None:
                position_dataset = h5f.require_dataset(
                    self._positions_key,
                    self._positions.shape,
                    dtype=np.float64,
                )
                position_dataset[...] = self._positions
                position_dataset.attrs["unit"] = "m"

            if self._origin_frames is not None:
                origin_frames_dataset = h5f.require_dataset(
                    self._origin_frames_key,
                    self._origin_frames.shape,
                    dtype=np.float64,
                )
                origin_frames_dataset[...] = self._origin_frames
                origin_frames_dataset.attrs["unit"] = "px"

            if self._camera_indices is not None:
                camera_indices_dataset = h5f.require_dataset(
                    self._camera_indices_key,
                    (len(self._camera_indices),),
                    dtype=np.int32,
                )

    @raise_if_outside_context
    def load(self):
        """
        Load data if exists
        If not, compute them from the trajectory data.
        """

        with h5py.File(self.path, "r") as h5f:
            if self._positions_key in h5f:
                self._positions = np.array(h5f[self._positions_key], dtype=np.float64)
            if self._origin_frames_key in h5f:
                self._origin_frames = np.array(h5f[self._origin_frames_key], dtype=np.float64)
            if self._camera_indices_key in h5f:
                self._camera_indices = list(h5f[self._camera_indices_key])

    def __enter__(self):
        """
        If file at self.path does not exist, create one.
        """
        self._inside_context = True
        assert os.path.exists(self.path), f"File does not exist {self.path}."
        self.load()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Save posture on the existing file
        """
        self.save()
        self._inside_context = False
