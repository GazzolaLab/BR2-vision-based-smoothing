"""
Created on Aug. 10, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import os
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as EmptyClass

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import br2_vision
from br2_vision.algorithms.frame_tools import rod_color
from br2_vision.algorithms.frames.frame_data import DirectorFrame
from br2_vision.algorithms.frames.frame_rod import RodFrame
from br2_vision.data_structure.marker_positions import MarkerPositions
from br2_vision.data_structure.posture_data import PostureData
from br2_vision.utility.logging import config_logging
from br2_vision_cli.run_smoothing import RequiredRawData, create_data_object

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

# from br2_vision.data_structure.smoothing import create_data_object, read_data_from_file


def rotate_frame(orientation, position=None, director=None):
    if position is not None:
        new_position = np.zeros(position.shape)
        for n in range(position.shape[1]):
            new_position[:, n] = orientation @ position[:, n]
        return new_position
    if director is not None:
        new_director = np.zeros(director.shape)
        for n in range(director.shape[2]):
            new_director[:, :, n] = director[:, :, n] @ orientation.T
        return new_director
    return None


class Frame(RodFrame, DirectorFrame):
    def __init__(
        self, figure_name, folder_name, fig_dict, gs_dict, rod_color, **kwargs
    ):
        RodFrame.__init__(
            self,
            figure_name,
            folder_name,
            fig_dict,
            gs_dict,
            rod_color=rod_color,
            **kwargs,
        )
        DirectorFrame.__init__(self, 0.02)

    def reset(
        self,
    ):
        RodFrame.reset(
            self,
        )
        DirectorFrame.reset(self, self.ax_rod, self.reference_length)


def create_movie(raw_data, file_path, delta_s_position, save_path):
    with open(file_path, "rb") as f:
        smoothed_data = pickle.load(f)
        # datafile_name = smoothed_data["datafile_name"]
        time = smoothed_data["time"]
        data_index = smoothed_data["data_index"]
        radius = smoothed_data["radius"]
        position = smoothed_data["position"]
        director = smoothed_data["director"]
        shear = smoothed_data["shear"]
        kappa = smoothed_data["kappa"]

    # raw_data = read_data_from_file(datafile_name)
    data, L0 = create_data_object(raw_data, delta_s_position)

    orientation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    # Create temporary directory
    with tempfile.TemporaryDirectory() as folder_name:
        frame = Frame(
            figure_name="frame{:04d}.png",
            folder_name=folder_name,
            fig_dict=dict(figsize=(18, 9)),
            gs_dict=dict(
                nrows=3,
                ncols=6,
                width_ratios=[1, 1, 1, 1, 1, 1],
                height_ratios=[1, 1, 1],
            ),
            rod_color="black",
            ax3d_flag=True,
        )

        rest_lengths = np.linalg.norm(position[0][:, 1:] - position[0][:, :-1], axis=0)
        frame.set_ref_configuration(
            position=rotate_frame(orientation, position=position[0]),
            shear=shear[0],
            kappa=kappa[0],
            reference_length=rest_lengths,
        )

        print("Plotting frames ...")
        for k in tqdm(range(len(time))):
            frame.reset()
            rod_ax = frame.plot_rod(
                position=rotate_frame(orientation, position=position[k]),
                director=rotate_frame(orientation, director=director[k]),
                radius=radius[k],
                color="black",
            )
            rod_ax.view_init(elev=10.0, azim=60)

            axes_shear, axes_curvature = frame.plot_strains(
                shear=shear[k], kappa=kappa[k]
            )

            if k != 0:
                frame.plot_data(
                    position=rotate_frame(
                        orientation, position=data.position[data_index[k]]
                    ),
                    director=rotate_frame(
                        orientation, director=data.director[data_index[k]]
                    ),
                )

            position_for_director = (position[k][:, 1:] + position[k][:, :-1]) / 2

            frame.plot_data(
                position=rotate_frame(
                    orientation, position=position_for_director[:, ::-5]
                ),
                director=rotate_frame(orientation, director=director[k][:, :, ::-5]),
                color="black",
            )

            frame.set_ax_rod_lim(
                z_lim=[-1.1, 0.1], x_lim=[-0.55, 0.55], y_lim=[-0.55, 0.55]
            )
            frame.set_ax_strains_lim(
                axes_shear_lim=[[-1.1, 1.1], [-1.1, 1.1], [-0.1, 2.1]],
                axes_curvature_lim=[
                    [-66, 66],
                    [-66, 66],
                    [-66, 66],
                ],
            )

            frame.set_labels(time[k])
            frame.save()

        frame.movie(frame_rate=30, movie_name=save_path)


@click.command()
@click.option(
    "-t",
    "--tag",
    type=str,
    help="Experiment tag. Path ./tag should exist.",
)
@click.option("-r", "--run_id", required=True, type=int, help="Run ID")
@click.option("-v", "--verbose", is_flag=True, type=bool, help="Verbose output")
def main(tag, run_id, verbose):
    config = br2_vision.load_config()
    config_logging(verbose)

    result_path = config["PATHS"]["results_dir"].format(tag, run_id)
    assert os.path.exists(result_path), "Run run_smoothing first"
    file_path = os.path.join(result_path, "smoothing.pkl")
    save_path = os.path.join(result_path, "smoothing.mov")
    # if problem == "bend":
    #     file_name = "bend"
    #     delta_s_position = np.array([23.9, 35.17, 33.82, 34.55, 32.8])
    # if problem == "twist":
    #     file_name = "bend"
    #     delta_s_position = np.array([23.9, 35.17, 33.82, 34.55, 32.8])
    # if problem == "mix":
    #     file_name = "mix"
    #     delta_s_position = np.array([23.9, 35.17, 33.82, 34.55, 32.8])
    # if problem == "cable":
    #     file_name = "cable"
    #     delta_s_position = np.array(
    #         [27.5, 33.5, 28, 34, 30, 31, 36.5, 31.5, 32, 34.5, 30, 31]
    #     )

    marker_positions = MarkerPositions.from_yaml(config["PATHS"]["marker_positions"])
    delta_s_position = np.array(marker_positions.marker_center_offset)

    # Create raw data
    raw_data = EmptyClass()
    tracing_data_path = config["PATHS"]["tracing_data_path"].format(tag, run_id)
    assert os.path.exists(
        tracing_data_path
    ), f"Tracing data does not exist: path={tracing_data_path}"
    with PostureData(path=tracing_data_path) as dataset:
        raw_data.time = dataset.get_time()
        raw_data.cross_section_center_position = (
            dataset.get_cross_section_center_position()
        )
        raw_data.cross_section_director = dataset.get_cross_section_director()

    create_movie(raw_data, file_path, delta_s_position, save_path)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Require problem keyword")
#     parser.add_argument(
#         "problem",
#         metavar="subproblem number",
#         type=str,
#         nargs=1,
#         help="problem keywork: bend, twist, mix",
#     )
#     args = parser.parse_args()
#     main(args.problem[0])
