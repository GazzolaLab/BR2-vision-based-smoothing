import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import yaml

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

    Attributes:
    -----------
    marker_center_offset: List[float]
        The relative interval of the markers along the rod.
    marker_positions: Dict[str, Tuple[float, float, float]]
        The relative position of the markers in cross-section.
    """

    marker_center_offset: List[float]
    marker_positions: Dict[str, Tuple[float, float, float]]

    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    marker_direction: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    normal_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self):
        """
        Convert marker_center_offset to list and marker_positions to dictionary.
        """
        self.marker_center_offset = [float(val) for val in self.marker_center_offset]
        self.marker_positions = {
            str(key): tuple(val) for key, val in self.marker_positions.items()
        }

    @property
    def tags(self):
        return list(self.marker_positions.keys())

    def get_total_count(self):
        """
        Get the total number of markers.
        """
        return len(self.marker_positions) * len(self.marker_center_offset)

    def get_count_per_ring(self):
        """
        Get the number of markers per plane.
        """
        return len(self.marker_center_offset)

    def __len__(self):
        """
        Get the number of plane.
        """
        return len(self.marker_positions)

    @property
    def Q(self):
        """
        Get the transformation matrix.
        """
        return np.stack(
            [
                self.normal_direction,
                np.cross(self.marker_direction, self.normal_direction),
                self.marker_direction,
            ]
        ).T

    def get_position(self, zidx: int, tag: str) -> np.ndarray:
        """
        Get the absolute position of a marker.
        """
        z_loc = np.cumsum(self.marker_center_offset)
        vec_marker = self.Q @ np.array(self.marker_positions[tag])
        z_position = z_loc[zidx] * np.array(self.marker_direction)
        return np.array(self.origin) + z_position + vec_marker

    @classmethod
    def from_h5(cls, path):
        """
        Load marker positions from h5 file.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"No h5 file to load marker position: {path}")

        with h5py.File(path, "r") as h5f:
            if "marker_positions" not in h5f:
                raise KeyError(f"marker_positions not in h5 file: {path}")
            grp: h5py.Group = h5f["marker_positions"]

            # Load data and create object
            _cls = cls(
                origin=tuple(grp.attrs["origin"]),
                marker_direction=tuple(grp.attrs["marker_direction"]),
                normal_direction=tuple(grp.attrs["normal_direction"]),
                marker_center_offset=list(grp["marker_center_offset"][()]),
                marker_positions=dict(
                    zip(
                        grp.attrs["tags"].astype(str),
                        [tuple(val) for val in grp["marker_positions"][()]],
                    )
                ),
            )
        return _cls

    def to_h5(self, path, overwrite=False):
        """
        Save marker positions to h5 file.

        If the group already exist in the file, the data will be overwritten based on the flag "overwrite".
        """
        self_data = asdict(self)
        with h5py.File(path, "a") as h5f:
            # If group exist and overwrite is False, skip.
            if overwrite or "marker_positions" not in h5f:
                if "marker_positions" in h5f:
                    grp = h5f["marker_positions"]
                else:
                    grp: h5py.Group = h5f.create_group("marker_positions")

                # Save data and attributes
                grp.attrs["origin"] = self_data["origin"]
                grp.attrs["marker_direction"] = self_data["marker_direction"]
                grp.attrs["normal_direction"] = self_data["normal_direction"]
                grp.attrs["tags"] = list(self.tags)  # , dtype="<U1")

                grp.create_dataset(
                    "marker_center_offset",
                    data=self_data["marker_center_offset"],
                    dtype=float,
                )
                grp.create_dataset(
                    "marker_positions",
                    data=np.array(list(self.marker_positions.values())),
                    dtype=float,
                )
