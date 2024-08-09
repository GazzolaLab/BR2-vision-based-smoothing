"""
Created on Aug. 01, 2021
@author: Heng-Sheng (Hanson) Chang
@modified by: Seung Hyun Kim
"""

import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as EmptyClass
from typing import Protocol

import click
import numpy as np
from elastica.rod.cosserat_rod import CosseratRod

import br2_vision
from br2_vision.algorithms.smoothing_algorithm import ForwardBackwardSmooth
from br2_vision.data_structure.marker_positions import MarkerPositions
from br2_vision.data_structure.posture_data import PostureData, SmoothingData
from br2_vision.utility.logging import config_logging, get_script_logger


class RequiredRawData(Protocol):
    @property
    def time(self) -> np.typing.NDArray[np.float64]: ...

    @property
    def cross_section_center_position(self) -> np.typing.NDArray[np.float64]: ...

    @property
    def cross_section_director(self) -> np.typing.NDArray[np.float64]: ...


def create_data_object(
    raw_data: RequiredRawData, delta_s_position: np.typing.NDArray[np.float64]
):
    data = EmptyClass()
    data.time = raw_data.time.copy()
    data.position = raw_data.cross_section_center_position.copy()
    data.director = raw_data.cross_section_director.copy()
    data.director_flag = True

    frame_index = 0
    data.noisy_position = data.position[frame_index, :, :].copy()
    data.noisy_director = data.director[frame_index, :, :, :].copy()

    blocksize = data.noisy_position.shape[1]

    # # delta_s_position = np.array([46.5, 41.25, 38, 35.5, 35, 48.5, 30, 32, 34.5])
    # # delta_s_position = np.array([27.5, 33.5, 28, 34, 30, 31, 36.5, 31.5, 32, 34.5, 30, 31])

    # delta_s_position = np.array([23.9, 35.17, 33.82, 34.55, 32.8])
    # delta_s_position = np.array([12, 32, 31, 29.5, 31.5, 30.5, 27, 30, 30, 31, 27, 31])

    # delta_s_position = np.array([12.0, 92.0, 89.0, 91.0])
    # delta_s_position = np.array([12, 63, 61, 57.5, 60, 58])

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

    data.s_position = np.cumsum(delta_s_position)
    L0 = data.s_position[-1]
    data.s_position /= data.s_position[-1]
    data.s_director = data.s_position.copy()
    return data, L0


def create_BR2(
    n=500, 
    L0=0.16, 
    radius=0.0075, 
    direction=np.array([0.0, 1.0, 0.0]),
    normal=np.array([1.0, 0.0, 0.0]),
    youngs_modulus=1e7,
):
    radii = radius * np.ones(n + 1)
    radii_mean = (radii[:-1] + radii[1:]) / 2
    rod = CosseratRod.straight_rod(
        n_elements=n,
        start=np.zeros((3,)),
        direction=direction,
        normal=normal,
        base_length=L0,
        base_radius=radii_mean.copy(),
        density=700,
        youngs_modulus=youngs_modulus,
    )
    return rod


def process_data(
    raw_data: RequiredRawData,
    markers: MarkerPositions,
    save_path: Path,
    h5_path: Path,
):

    data, L0 = create_data_object(
        raw_data=raw_data, 
        delta_s_position=np.array(markers.marker_center_offset)
    )

    # define rod
    n_elem = 100
    radius = 0.0075 * 2.742
    rod = create_BR2(
        n=n_elem, 
        L0=L0, 
        radius=radius,
        direction=np.array(markers.marker_direction),
        normal=np.array(markers.normal_direction),
    )

    # define smoothing algorithm
    algo_config = EmptyClass()
    algo_config.argument_weight = 1
    algo_config.step_size = 1e-6
    algo_config.data_deviation_weight_cost = 1_000_000
    algo_config.data_deviation_weight_cost_position = algo_config.data_deviation_weight_cost
    algo_config.data_deviation_weight_cost_director = algo_config.data_deviation_weight_cost

    # run smoothing algorithm
    algo = ForwardBackwardSmooth(rod, algo_config, data)

    smoothed_data = defaultdict(list)
    smoothed_data["time"].append(0)
    smoothed_data["data_index"].append(None)
    smoothed_data["radius"].append(algo.radius.copy())
    smoothed_data["position"].append(algo.position.copy())
    smoothed_data["director"].append(algo.director.copy())
    smoothed_data["shear"].append(algo.shear.copy())
    smoothed_data["kappa"].append(algo.kappa.copy())

    for k in range(data.time.shape[0] - 1):
        algo.data.noisy_position = data.position[k, :, :].copy()
        algo.data.noisy_director = data.director[k, :, :, :].copy()

        # Run
        stime = time.time()
        cost = algo.run(
            iter_number=100_000, 
            threshold=1e-5,
            cost_threshold=algo_config.data_deviation_weight_cost*10,
        )
        runtime = time.time() - stime

        # Report delta
        last_cost = cost[-1]/algo_config.data_deviation_weight_cost
        delta_cost = np.abs((cost[-1] - cost[-2]) / cost[-2])
        delta_position = (
            np.abs(smoothed_data["position"][-1] - algo.position).max() * 1000
        )
        delta_director = np.abs(smoothed_data["director"][-1] - algo.director).max()
        delta_shear = np.abs(smoothed_data["shear"][-1] - algo.shear).max()
        delta_kappa = np.abs(smoothed_data["kappa"][-1] - algo.kappa).max()
        num_iter = len(cost)
        print(
            f"time step {k=} ({data.time[k]=:0.2e}) : {runtime=:0.2e} sec, {last_cost=:0.2e}, {delta_cost=:0.2e}, {delta_position=:0.2f} mm, {delta_director=:0.2e}, {delta_shear=:0.2e}, {delta_kappa=:0.2e}, {num_iter=}"
        )

        # Save
        smoothed_data["time"].append(data.time[k])
        smoothed_data["data_index"].append(k)
        smoothed_data["radius"].append(algo.radius.copy())
        smoothed_data["position"].append(algo.position.copy())
        smoothed_data["director"].append(algo.director.copy())
        smoothed_data["shear"].append(algo.shear.copy())
        smoothed_data["kappa"].append(algo.kappa.copy())

    # Export
    print("Saving data to pickle files ...", end="\r")

    with open(save_path, "wb") as data_file:
        data = dict(
            time=smoothed_data["time"],
            data_index=smoothed_data["data_index"],
            radius=smoothed_data["radius"],
            position=smoothed_data["position"],
            director=smoothed_data["director"],
            shear=smoothed_data["shear"],
            kappa=smoothed_data["kappa"],
        )
        pickle.dump(data, data_file)

    with SmoothingData(h5_path) as data:
        data.set(
            time=np.array(smoothed_data["time"]),
            data_index=np.array(smoothed_data["data_index"]),
            radius=np.array(smoothed_data["radius"]),
            position=np.array(smoothed_data["position"]),
            director=np.array(smoothed_data["director"]),
            shear=np.array(smoothed_data["shear"]),
            kappa=np.array(smoothed_data["kappa"]),
        )


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
    logger = get_script_logger(os.path.basename(__file__))

    save_path = config["PATHS"]["results_dir"].format(tag, run_id)
    os.makedirs(save_path, exist_ok=True)

    markers = MarkerPositions.from_yaml(config["PATHS"]["marker_positions"])

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

    process_data(
        raw_data,
        markers,
        os.path.join(save_path, "smoothing.pkl"),
        h5_path=tracing_data_path,
    )
