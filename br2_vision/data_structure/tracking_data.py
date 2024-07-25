import operator
import os
from dataclasses import dataclass
from typing import List, Tuple, Union

import h5py
import numpy as np
from nptyping import Floating, NDArray, Shape

from br2_vision.utility.logging import get_script_logger

from .marker_positions import MarkerPositions
from .utils import raise_if_outside_context


def h5_track_dir(z_index: "int", label: "str"):
    path = "/dlt-track"
    path += f"/z_{z_index}"
    path += f"/label_{label}"
    return path


def h5_directory(
    camera: "int|None" = None, z_index: "int|None" = None, label: "str|None" = None
):
    path = "/trajectory"
    if camera is None:
        return path
    path += f"/camera_{camera}"
    if z_index is None:
        return path
    path += f"/z_{z_index}"
    if label is None:
        return path
    path += f"/label_{label}"
    return path


def compose_tag(z_index, label):
    return f"z{z_index}-{label}"


@dataclass
class FlowQueue:
    """
    Data structure for storing flow queue
    Fixed parameters: point, camera, z_index, label
    Adjustable parameters: start_frame, end_frame, done

    Parameters:
    - point: (x, y) pixel coordinates
    - start_frame: start frame of the trajectory in video frame
    - end_frame: end frame of the trajectory in video frame
    - camera: camera id
    - z_index: z-index of the marker along the rod
    - label: label of the marker
    - done: flag to indicate if the flow queue has been processed
    """

    point: Tuple[int, int]
    start_frame: int
    end_frame: int
    camera: int
    z_index: int
    label: str
    done: bool = False

    dtype = [
        ("point", "i4", (2,)),
        ("start_frame", "<i4"),
        ("end_frame", "<i4"),
        ("camera", "<i4"),
        ("z_index", "<i4"),
        ("label", "S10"),
        ("done", "?"),
    ]

    __initialized__ = False
    __static_variables__ = ["point", "camera", "z_index", "label"]

    def __setattr__(self, name, value):
        # type check
        if name == "point":
            if not isinstance(value, tuple):
                raise TypeError(
                    f"Expected tuple, got {type(value)} for parameter {name}"
                )
            assert (
                len(value) == 2
            ), f"Expected length 2, got {len(value)} for parameter {name}"
            if not all([isinstance(val, int) for val in value]):
                raise TypeError(
                    f"Expected int, got {[type(val) for val in value]} for parameter {name}"
                )
        elif name in ["start_frame", "end_frame", "camera", "z_index"]:
            if not isinstance(value, int):
                raise TypeError(f"Expected int, got {type(value)} for parameter {name}")
        elif name == "label":
            if not isinstance(value, str):
                raise TypeError(f"Expected str, got {type(value)} for parameter {name}")
        elif name == "done":
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected bool, got {type(value)} for parameter {name}"
                )
        super().__setattr__(name, value)
        if self.__initialized__:
            if name in self.__static_variables__:
                raise AttributeError(f"Cannot change {name} after initialization")

    def __post_init__(self):
        self.__initialized__ = True

    def __eq__(self, other):
        return all(
            [
                self.point[0] == other.point[0],
                self.point[1] == other.point[1],
                self.start_frame == other.start_frame,
                self.end_frame == other.end_frame,
                self.camera == other.camera,
                self.z_index == other.z_index,
                self.label == other.label,
            ]
        )

    def __array__(self, dtype=None) -> np.ndarray:
        val = np.recarray((1,), dtype=self.dtype)
        val.point = self.point
        val.start_frame = self.start_frame
        val.end_frame = self.end_frame
        val.camera = self.camera
        val.z_index = self.z_index
        val.label = self.label
        val.done = self.done
        return val

    @property
    def h5_directory(self):
        return h5_directory(self.camera, self.z_index, self.label)

    def get_tag(self):
        z_index = self.z_index  # .decode("utf-8")
        label = self.label  # .decode("utf-8")
        return compose_tag(z_index, label)


class TrackingData:
    """
    Data structure for storing tracking data
    """

    def __init__(self, path, marker_positions: MarkerPositions):
        self.queues: List[FlowQueue] = []
        self.path = path
        self.marker_positions = marker_positions

        self.logger = get_script_logger(os.path.basename(__file__))

        self._inside_context = False

    @property
    @raise_if_outside_context
    def all_done(self):
        if len(self.queues) == 0:
            return True
        return all([q.done for q in self.queues])

    @raise_if_outside_context
    def iter_cameras(self) -> list[int]:
        # Get unique camera id
        cameras = set([q.camera for q in self.queues])
        # return sorted list
        return sorted(cameras)

    @raise_if_outside_context
    def save_pixel_flow_trajectory(
        self,
        data: np.ndarray,
        flow_queue: FlowQueue,
        size: "int | None" = None,
        dim: int = 2,
        prefix="xy",
        full_trajectory=False,
    ):
        """
        Save trajectory in h5 file
        - Create (if doesn't exist) directory: /trajectory/camera_{cid}/z_{z_index}/label_{label}
        - Save data in the directory

        Shape of data is expected to be (N, dim).
        N is the number of frames for the flow queue.
        If full_trajectory is True, N is the total number of frames in the video.

        Parameters:
        -----------
        data: np.ndarray
            Trajectory data
        flow_queue: FlowQueue
        size: int
            Full size. (most likely the number of frames in the video)
        dim: int
            Dimension of the data. (default: 2)
        prefix: str
            (default: "xy")
        full_trajectory: bool
            Save full trajectory if True, otherwise save only the trajectory between start_frame and end_frame.
            (default: False)
        """
        # Create directory
        with h5py.File(self.path, "a") as h5f:
            # Check if directory exists
            grp = h5f.require_group(flow_queue.h5_directory)
            if prefix in grp:
                dset = grp[prefix]
                if size is not None:
                    assert (
                        dset.shape[0] == size
                    ), f"Size mismatch: {dset.shape[0]} != {size}"
            else:
                # initialize dataset with nan
                assert (
                    size is not None
                ), "Trajectory size should be provided if dataset label does not exist."
                shape = (size, dim)
                dset = grp.create_dataset(
                    prefix,
                    shape,
                    dtype=np.int_,
                    data=np.full(shape, -1, dtype=np.int_),
                )
                dset.attrs["unit"] = "pixel"

            if full_trajectory:
                assert (
                    dset.shape == data.shape
                ), f"Shape mismatch: {dset.shape} != {data.shape}"
                dset[...] = data
            else:
                assert (
                    flow_queue.end_frame - flow_queue.start_frame == data.shape[0]
                ), f"Shape mismatch: nframes:{flow_queue.end_frame - flow_queue.start_frame} != ndata:{data.shape[0]}"
                dset[flow_queue.start_frame : flow_queue.end_frame] = data

    @raise_if_outside_context
    def load_pixel_flow_trajectory(
        self, flow_queue: FlowQueue, prefix="xy", full_trajectory=False
    ):
        """
        Load trajectory from h5 file
        """
        with h5py.File(self.path, "r") as h5f:
            if flow_queue.h5_directory not in h5f:
                return None
            grp = h5f[flow_queue.h5_directory]
            dset = grp[prefix]
            if full_trajectory:
                return np.array(dset[:], dtype=np.int_)
            else:
                return np.array(
                    dset[flow_queue.start_frame : flow_queue.end_frame], dtype=np.int_
                )

    @raise_if_outside_context
    def save_timestamps(self, timestamps):
        """
        Save timestamps in h5 file
        """
        with h5py.File(self.path, "a") as h5f:
            dset = h5f.require_dataset(
                "dlt-track/timestamps",
                timestamps.shape,
                dtype=np.float64,
                data=timestamps,
            )
            dset.attrs["unit"] = "s"

        print(f"Saved timestamps size {timestamps.shape} to /dlt-track/timestamps")

    @raise_if_outside_context
    def load_track(
        self,
        z_index: int,
        label: str,
        prefix: str = "xyz",
    ) -> "NDArray[Shape['T', 'D'], Floating] | None":
        """
        Load trajectory from h5 file
        """
        with h5py.File(self.path, "r") as h5f:
            directory = h5_track_dir(z_index, label)
            if directory not in h5f:
                return None
            grp = h5f[directory]
            dset = grp[prefix]
            data = np.array(dset[:], dtype=np.float64)
        return data

    @raise_if_outside_context
    def save_track(
        self, data, z_index: int, label: str, prefix: str = "xyz", unit: str = "m"
    ):
        """
        Load trajectory from h5 file

        Parameters:
        -----------
        data: np.ndarray[np.floating]
            Trajectory data
        camera_id: int
        z_index: int
        label: str
        prefix: str
            (default: "xy")
        """

        with h5py.File(self.path, "a") as h5f:
            directory = h5_track_dir(z_index, label)
            grp = h5f.require_group(directory)
            if prefix in grp:
                dset = grp[prefix]
                assert (
                    dset.shape == data.shape
                ), f"Shape mismatch: {dset.shape} != {data.shape}"
            else:
                dset = grp.create_dataset(
                    prefix,
                    data.shape,
                    dtype=np.float64,
                    data=data,
                )
                dset.attrs["unit"] = unit
            data = np.array(dset[:], dtype=np.float64)

        print(
            f"Saved track z{z_index}:{label} size {data.shape} to {directory} - prefix-{prefix}"
        )

    @raise_if_outside_context
    def load_trajectory(
        self,
        camera_id: int,
        z_index: int,
        label: str,
        prefix="xy",
    ):
        """
        Load trajectory from h5 file
        """
        with h5py.File(self.path, "r") as h5f:
            directory = h5_directory(camera_id, z_index, label)
            if directory not in h5f:
                return None
            grp = h5f[directory]
            dset = grp[prefix]
            return np.array(dset[:], dtype=np.int_)

    @raise_if_outside_context
    def query_timelength(
        self,
        prefix="xy",
    ) -> int:
        """
        Get the length of the smallest trajectory
        """
        lengths = None
        for cam_id in self.iter_cameras():
            with h5py.File(self.path, "r") as h5f:
                directory = h5_directory(cam_id)
                grp = h5f[directory]
                dset = grp[list(grp.keys())[0]]
                data = dset[list(dset.keys())[0]][prefix]
                lengths = (
                    min(lengths, data.shape[0])
                    if lengths is not None
                    else data.shape[0]
                )

        return lengths

    @raise_if_outside_context
    def query_trajectory(
        self,
        z_index: int,
        label: str,
        timelength: "int|None" = None,
        prefix="xy",
    ):
        """
        Load trajectory from h5 file
        """
        point_collections = {}
        for cam_id in self.iter_cameras():
            data = self.load_trajectory(cam_id, z_index, label, prefix=prefix)
            if data is None:
                continue
            if timelength is not None:
                point_collections[cam_id] = data[:timelength]
            else:
                point_collections[cam_id] = data
        return point_collections

    @raise_if_outside_context
    def trim_trajectory(
        self,
        tag: str,
        frame: int,
        prefix="xy",
        reverse=False,
    ):
        """
        Trim trajectory beyond the frame
        """

        # find queue with matching tag
        for q in self.queues:
            if q.get_tag() == tag and frame >= q.start_frame and frame <= q.end_frame:
                # load trajectory
                relative_frame = frame - q.start_frame
                trajectory = self.load_pixel_flow_trajectory(
                    q,
                    prefix=prefix,
                )
                if reverse:
                    trajectory[:relative_frame] = -1
                    self.save_pixel_flow_trajectory(
                        trajectory,
                        q,
                        prefix=prefix,
                    )
                    q.start_frame = frame
                else:
                    trajectory[relative_frame:] = -1
                    self.save_pixel_flow_trajectory(
                        trajectory,
                        q,
                        prefix=prefix,
                    )
                    q.end_frame = frame

    @classmethod
    def initialize(cls, path, marker_positions):
        """
        Initialize the tracking data object.
        """
        if os.path.exists(path):
            return cls.load(path)
        return cls(path, marker_positions)

    @classmethod
    def load(cls, path):
        assert os.path.exists(path), f"File does not exist {path}."
        marker_positions = MarkerPositions.from_h5(path)
        with h5py.File(path, "r") as h5f:
            # Load queues
            dset = h5f["queues"]
            queues = []
            for vals in dset[...].tolist():
                # Convert to FlowQueue datatype
                vals = list(vals)
                vals[0] = tuple(int(vals[0][i]) for i in range(2))
                vals[5] = vals[5].decode()
                fq = FlowQueue(*vals)
                queues.append(fq)
        # Reset parameters
        c = cls(path, marker_positions=marker_positions)
        c.queues = queues
        return c

    @raise_if_outside_context
    def create_template(self):
        """
        Initialize data structure.
        """
        # create directory if necessary
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with h5py.File(self.path, "w") as h5f:
            dset = h5f.create_dataset(
                "queues",
                (1,),
                maxshape=(None,),
                dtype=FlowQueue.dtype,
            )
        self.marker_positions.to_h5(self.path)

    def __enter__(self):
        """
        If file at self.path does not exist, create one.
        """
        self._inside_context = True
        if not os.path.exists(self.path):
            self.create_template()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Save queue on the existing file
        """
        with h5py.File(self.path, "a") as h5f:
            dset = h5f["queues"]
            dset.resize((len(self.queues),))
            for idx, q in enumerate(self.queues):
                dset[idx] = np.array(q)
        self.marker_positions.to_h5(self.path, overwrite=True)
        self._inside_context = False

    @raise_if_outside_context
    def append(self, value: FlowQueue):
        # if same value is already in the list, replace values
        if value in self.queues:
            idx = self.queues.index(value)
            self.queues[idx] = value
        else:
            self.queues.append(value)

    @raise_if_outside_context
    def get_flow_queues(
        self, camera=None, start_frame=None, force_run_all: bool = False, tag=None
    ) -> list[FlowQueue]:
        """
        General filter method
        """
        ret = []
        for queue in self.queues:
            # Filter
            if camera is not None and queue.camera != camera:
                continue
            if start_frame is not None and queue.start_frame != start_frame:
                continue
            if tag is not None and queue.get_tag() != tag:
                continue

            # Skip already-done queues
            if queue.done and not force_run_all:
                continue
            ret.append(queue)
        return ret
