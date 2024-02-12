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

    origin: Tuple[float, float, float] = (0., 0., 0.)
    marker_direction: Tuple[float, float, float] = (0., 1., 0.)

    @property
    def dtype(self):
        """
        dtype compatible for h5py
        """
        return [
            ('origin', float),
            ('marker_direction', float),
            ('marker_center_offset', float),
            ('marker_positions', float, (3,)),
            ('tags', 'S50')
        ]

    @property
    def tags(self):
        return self.marker_positions.keys()

    def __len__(self):
        return len(self.marker_center_location)

    def get_position(self, zidx: int, tag: str):
        """
        Get the absolute position of a marker.
        """
        raise NotImplementedError
