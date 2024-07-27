"""
Created on Aug. 02, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from br2_vision.algorithms.frame_tools import default_colors

# def include_parent_folders(parent_folders):
#     for parent_folder in parent_folders:
#         path = os.path.abspath(__file__)
#         for directory in path.split("/")[::-1]:
#             if directory == parent_folder:
#                 break
#             path = os.path.dirname(path)
#         sys.path.append(path)


# include_parent_folders(
#     parent_folders=[
#         "elastica-python",
#         "Smoothing",
#     ]
# )


class PositionFrame(object):
    def __init__(self):
        pass

    def reset(self, ax, reference_length):
        self.ax_data_frame = ax
        self.reference_length = reference_length

    def plot_data(self, position, color=None):
        blocksize = position.shape[1]
        for n in range(blocksize):
            color_index = n % len(default_colors)
            self.ax_data_frame.scatter(
                position[0, n] / self.reference_length,
                position[1, n] / self.reference_length,
                position[2, n] / self.reference_length,
                color=default_colors[color_index] if color is None else color,
            )


class DirectorFrame(PositionFrame):
    def __init__(self, director_scale):
        PositionFrame.__init__(
            self,
        )
        self.director_scale = director_scale

    def reset(self, ax, reference_length):
        PositionFrame.reset(self, ax, reference_length)

    def plot_data(self, position, director=None, color=None):
        PositionFrame.plot_data(self, position, color)
        if director is not None:
            blocksize = position.shape[1]
            for n in range(blocksize):
                color_index = n % len(default_colors)
                for i in range(3):
                    positions = np.zeros((3, 2))
                    positions[:, 0] = position[:, n]
                    positions[:, 1] = (
                        positions[:, 0] + director[i, :, n] * self.director_scale
                    )
                    self.ax_data_frame.plot(
                        positions[0] / self.reference_length,
                        positions[1] / self.reference_length,
                        positions[2] / self.reference_length,
                        color=default_colors[color_index] if color is None else color,
                    )
