from typing import NamedTuple, Tuple, List

import os
import numpy as np
import h5py

from .marker_positions import MarkerPositions


class FlowQueue(NamedTuple):
    tag: str
    point: Tuple[int, int]
    start_frame: int
    end_frame: int
    camera: int

    dtype = [
        ("tag", "S50"),
        ("point", int, (2,)),
        ("start_frame", int),
        ("end_frame", int),
        ("camera", int),
    ]


# string_names = ['Paul', 'John', 'Anna']
# float_heights = [5.9, 5.7,  6.1]
# int_ages = [27, 31, 33]
# numpy_data = [ np.array([5.4, 6.7, 8.8]),
#                np.array([3.1, 58.4, 66.4]),
#                np.array([4.7, 5.1, 4.2])  ]
#
# # Create empty record array with 3 rows
# ds_dtype = [('name','S50'), ('height',float), ('ages',int), ('numpy_data', float, (3,) ) ]
# ds_arr = np.recarray((3,),dtype=ds_dtype)
# # load list data to record array by field name
# ds_arr['name'] = np.asarray(string_names)
# ds_arr['height'] = np.asarray(float_heights)
# ds_arr['ages'] = np.asarray(int_ages)
# ds_arr['numpy_data'] = np.asarray(numpy_data)
#
# with h5py.File('SO_59483094.h5', 'w') as h5f:
# # load data to dataset my_ds1 using recarray
#     dset = h5f.create_dataset('my_ds1', data=ds_arr, maxshape=(None) )
# # load data to dataset my_ds2 by lists/field names
#     dset = h5f.create_dataset('my_ds2', dtype=ds_dtype, shape=(100,), maxshape=(None) )
#     dset['name',0:3] = np.asarray(string_names)
#     dset['height',0:3] = np.asarray(float_heights)
#     dset['ages',0:3] = np.asarray(int_ages)
#     dset['numpy_data',0:3] = np.asarray(numpy_data)

# Tracking data:
class TrackingData:
    def __init__(self, path, marker_positions: MarkerPositions):
        self.queues: List[FlowQueue] = []
        self.path = path
        self.marker_positions = marker_positions

    @classmethod
    def initialize(cls, path, marker_positions):
        return cls(path, marker_positions)

    @classmethod
    def load(cls, path):
        assert os.path.exists(path), "File does not exist."
        marker_positions = MarkerPositions.from_h5(path)
        with h5py.File(path, "r") as h5f:
            # Load queues
            dset = h5f["queue"]
            queues = dset[...]

        c = cls(path, marker_positions=marker_positions)
        c.queues = queues
        return c

    def create_template(self):
        """
        Data Structure:
        """
        with h5py.File(self.path, "w") as h5f:
            dset = h5f.create_dataset("queues", (0,), dtype=FlowQueue.dtype)
        self.marker_positions.to_h5(self.path)

    def __enter__(self):
        """
        If file at self.path does not exist, create one.
        """
        if not os.path.exists(self.path):
            self.create_template()

    def __exit__(self):
        """
        Save queue on the existing file
        """
        with h5py.File(self.path, "a") as h5f:
            dset = h5f["queues"]
            dset.resize((len(self.queues),))
            dset[...] = self.queues

    def append(self, value):
        self.queues.append(value)

    def get_flow_queues(self, camera=None, start_frame=None):
        ret = []
        for queue in self.queues:
            if camera is not None and queue.camera != camera:
                continue
            if start_frame is not None and queue.start_frame < start_frame:
                continue
            ret.append(queue)
        return ret
