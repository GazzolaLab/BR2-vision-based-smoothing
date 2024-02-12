from typing import Dict, Tuple

import os, sys
import yaml
import h5py
import numpy as np

from pathlib import Path
import dataclasses
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class MarkerPositions:
    """
    Class to load and store marker positions.
    `Marker` is a set of points in 3D space which is used to determine the center-position (0,0,0) and the orientation (ex, ey, ez) of the cross-section.

    The class offers methods to load and store marker positions.
    The information can be exported or loaded from yaml files.
    The yaml files consist of simple dictionary with the marker id as key and the coordinates as value.

    The class offers methods to convert positions from relative to absolute coordinates, and a method to find orientation for each marker.
    """

    marker_location: List[float]
    marker_positions: Dict[str, Tuple[float, float, float]]

    @classmethod
    def from_yaml(cls, file_path: str):
        """
        Load current dataclass from a yaml file.
        """
        with open(file_path, "r") as file:
            data_dict = yaml.safe_load(file)
        return cls(**data_dict)

    def to_yaml(self, file_path: str):
        """
        Save current dataclass from a yaml file.
        """
        data_dict = dataclasses.asdict(self, dict_factory=OrderedDict)
        with open(file_path, "w") as file:
            yaml.dump(self, file)
