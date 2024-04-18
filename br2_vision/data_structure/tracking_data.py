from typing import Tuple, List

import os
from dataclasses import dataclass
import operator

import numpy as np
import h5py

from br2_vision.utility.logging import get_script_logger

from .marker_positions import MarkerPositions


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
                raise TypeError(f"Expected tuple, got {type(value)} for parameter {name}")
            assert len(value) == 2, f"Expected length 2, got {len(value)} for parameter {name}"
            if not all([isinstance(val, int) for val in value]):
                raise TypeError(f"Expected int, got {[type(val) for val in value]} for parameter {name}")
        elif name in ["start_frame", "end_frame", "camera", "z_index"]:
            if not isinstance(value, int):
                raise TypeError(f"Expected int, got {type(value)} for parameter {name}")
        elif name == "label":
            if not isinstance(value, str):
                raise TypeError(f"Expected str, got {type(value)} for parameter {name}")
        elif name == "done":
            if not isinstance(value, bool):
                raise TypeError(f"Expected bool, got {type(value)} for parameter {name}")
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
        return f"/trajectory/camera_{self.camera}/z_{self.z_index}/label_{self.label}"

    def get_tag(self):
        z_index = self.z_index  #.decode("utf-8")
        label = self.label  #.decode("utf-8")
        return f"z{z_index}-{label}"

def raise_if_outside_context(method):  # pragma: no cover
    def decorator(self, *args, **kwargs):
        if not self._inside_context:
            raise Exception("This method should be called from inside context.")
        return method(self, *args, **kwargs)
    return decorator

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
    def all_done(self):
        return all([q.done for q in self.queues])

    def iter_cameras(self):
        # Get unique camera id
        cameras = set([q.camera for q in self.queues])
        # return sorted list
        return sorted(cameras)

    def save_pixel_flow_trajectory(
        self,
        data: np.ndarray,
        flow_queue: FlowQueue,
        size: int,
        prefix="xy",
        full_trajectory=False,
    ):
        """
        Save trajectory in h5 file
        - Create (if doesn't exist) directory: /trajectory/camera_{cid}/z_{z_index}/label_{label}
        - Save data in the directory
        """
        # Create directory
        with h5py.File(self.path, "a") as h5f:
            # Check if directory exists
            grp = h5f.require_group(flow_queue.h5_directory)
            if prefix in grp:
                dset = grp[prefix]
                assert (
                    dset.shape[0] == size
                ), f"Data shape mismatch: {dset.shape[0]} != {size}"
            else:
                # initialize dataset with nan
                shape = (size, 2)
                dset = grp.create_dataset(
                    prefix,
                    shape,
                    dtype=np.int_,
                    data=np.full(shape, -1, dtype=np.int_),
                )
                dset.attrs["unit"] = "pixel"

            if full_trajectory:
                dset[...] = data
            else:
                dset[flow_queue.start_frame : flow_queue.end_frame] = data

    def load_pixel_flow_trajectory(
        self, flow_queue: FlowQueue, prefix="xy", full_trajectory=False
    ):
        """
        Load trajectory from h5 file
        """
        with h5py.File(self.path, "r") as h5f:
            grp = h5f[flow_queue.h5_directory]
            dset = grp[prefix]
            if full_trajectory:
                return np.array(dset, dtype=np.int_)
            else:
                return dset[flow_queue.start_frame : flow_queue.end_frame]

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
            if q.get_tag() == tag:
                # set end-frame to be the frame
                q.end_frame = frame

                # load trajectory
                trajectory = self.load_pixel_flow_trajectory(
                    q, prefix=prefix, full_trajectory=True
                )
                if reverse:
                    trajectory[:frame] = -1
                else:
                    trajectory[frame:] = -1
                self.save_pixel_flow_trajectory(
                    trajectory, q, len(trajectory), prefix=prefix, full_trajectory=True
                )

    @classmethod
    def initialize(cls, path, marker_positions):
        """
        Initialize the tracking data object. If file exists, raise error.
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
                vals[6] = str(vals[6], encoding="utf-8")
                fq = FlowQueue(*vals)
                queues.append(fq)

        c = cls(path, marker_positions=marker_positions)
        c.queues = queues
        return c

    def create_template(self):
        """
        Initialize data structure.
        """
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
        if not os.path.exists(self.path):
            self.create_template()
        self._inside_context = True
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
        self._inside_context = False

    def append(self, value):
        # if same value is already in the list, replace values
        if value in self.queues:
            idx = self.queues.index(value)
            self.queues[idx] = value
        else:
            self.queues.append(value)

    def get_flow_queues(
        self, camera=None, start_frame=None, force_run_all: bool = False, tag=None
    ):
        """
        General filter method
        """
        ret = []
        for queue in self.queues:
            # Filter
            if camera is not None and queue.camera != camera:
                continue
            if start_frame is not None and queue.start_frame < start_frame:
                continue
            if tag is not None and queue.get_tag() != tag:
                continue

            # Skip already-done queues
            if queue.done and not force_run_all:
                continue
            ret.append(queue)
        return ret
