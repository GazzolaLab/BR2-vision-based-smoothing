from typing import Tuple, List

import os
from dataclasses import dataclass
import numpy as np
import h5py


from br2_vision.utility.logging import get_script_logger

from .marker_positions import MarkerPositions


@dataclass
class FlowQueue:
    point: Tuple[int, int]
    start_frame: int
    end_frame: int
    camera: int
    z_index: str
    label: str
    done: bool = False

    dtype = [
        ("point", "i8", (2,)),
        ("start_frame", "<i8"),
        ("end_frame", "<i8"),
        ("camera", "<i8"),
        ("z_index", "S10"),
        ("label", "S10"),
        ("done", "?"),
    ]

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

    def h5_directory(self):
        return f"/trajectory/camera_{self.camera}/z_{self.z_index}/label_{self.label}"

    def get_tag(self):
        z_index = self.z_index.decode("utf-8")
        label = self.label.decode("utf-8")
        return f"z{z_index}-{label}"


class TrackingData:
    def __init__(self, path, marker_positions: MarkerPositions):
        self.queues: List[FlowQueue] = []
        self.path = path
        self.marker_positions = marker_positions

        self.logger = get_script_logger(os.path.basename(__file__))

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
            directory = flow_queue.h5_directory()
            grp = h5f.require_group(directory)
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
        flow_queue.done = True

    def load_pixel_flow_trajectory(
        self, flow_queue: FlowQueue, prefix="xy", full_trajectory=False
    ):
        """
        Load trajectory from h5 file
        """
        with h5py.File(self.path, "r") as h5f:
            directory = flow_queue.h5_directory()
            grp = h5f[directory]
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
        # find queue with matching tag
        for q in self.queues:
            if q.get_tag() == tag:
                break

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
        return cls(path, marker_positions)

    @classmethod
    def load(cls, path):
        assert os.path.exists(path), f"File does not exist {path}."
        marker_positions = MarkerPositions.from_h5(path)
        with h5py.File(path, "r") as h5f:
            # Load queues
            dset = h5f["queues"]
            queues = [FlowQueue(*vals) for vals in dset[...].tolist()]

        c = cls(path, marker_positions=marker_positions)
        c.queues = queues
        return c

    def create_template(self):
        """
        Data Structure:
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

    def append(self, value):
        self.queues.append(value)

    def get_flow_queues(
        self, camera=None, start_frame=None, force_run_all: bool = False
    ):
        ret = []
        for queue in self.queues:
            # Filter
            if camera is not None and queue.camera != camera:
                continue
            if start_frame is not None and queue.start_frame < start_frame:
                continue

            # Skip already-done queues
            if queue.done and not force_run_all:
                continue
            ret.append(queue)
        return ret
