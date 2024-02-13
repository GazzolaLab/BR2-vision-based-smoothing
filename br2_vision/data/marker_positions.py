from typing import Dict, Tuple, List

import os, sys
import yaml
import h5py
import numpy as np

from pathlib import Path
from dataclasses import dataclass

from .io_utils import DataclassYamlSaveLoadMixin


@dataclass
class MarkerPositions(DataclassYamlSaveLoadMixin):
    """
    Class to load and store marker positions.
    Assume markers are along a straight line in 3d.
    `Marker` is a set of points in 3D space which is used to determine the center-position (0,0,0) and the orientation (ex, ey, ez) of the cross-section.

    The class offers methods to load and store marker positions.
    The information can be exported or loaded from yaml files.
    The yaml files consist of simple dictionary with the marker id as key and the coordinates as value.

    The class offers methods to convert positions from relative to absolute coordinates, and a method to find orientation for each marker.

    Access the absolute location by calling `get_position(location index(int), position id(str))`.

    Mixin - DataclassYamlSaveLoadMixin provides `from_yaml` and `to_yaml` methods to load and save.
    """

    marker_center_offset: List[float]
    marker_positions: Dict[str, Tuple[float, float, float]]

    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    marker_direction: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    dtype = [
        ("origin", float),
        ("marker_direction", float),
        ("marker_center_offset", float),
        ("marker_positions", float, (3,)),
        ("tags", "S50"),
    ]

    @property
    def tags(self):
        return self.marker_positions.keys()

    def get_total_count(self):
        """
        Get the total number of markers.
        """
        return len(self.marker_positions) * len(self.marker_center_offset)

    def get_count_per_plane(self):
        """
        Get the number of markers per plane.
        """
        return len(self.marker_center_offset)

    def __len__(self):
        """
        Get the number of plane.
        """
        return len(self.marker_center_location)

    def get_position(self, zidx: int, tag: str):
        """
        Get the absolute position of a marker.
        """
        raise NotImplementedError

    def from_h5(self, path):
        """
        Load marker positions from h5 file.
        """
        with h5py.File(path, "r") as h5f:
            dset = h5f["marker_positions"]
            return dset[...]

    def to_h5(self, path):
        """
        Save marker positions to h5 file.
        """
        with h5py.File(path, "w") as h5f:
            dset = h5f.create_dataset("marker_positions", (0,), dtype=MarkerPositions.dtype, data=self)
