"""
Created on Aug. 01, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from collections import defaultdict

from tqdm import tqdm
import time

import numpy as np
import pickle

from types import SimpleNamespace as EmptyClass

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

import os, sys

# os.environ['NUMBA_DISABLE_JIT'] = '1'

from elastica.rod.cosserat_rod import CosseratRod

from algorithms.smoothing_algorithm import ForwardBackwardSmooth


def read_data_from_file(file_name):
    raw_data = np.load("data/" + file_name + ".npz")
    return raw_data


def create_data_object(raw_data, delta_s_position):
    data = EmptyClass()
    data.time = raw_data["time"].copy()
    data.position = raw_data["cross_section_center_position"].copy()

    data.director = raw_data["cross_section_director"].copy()
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

    data.s_position = np.cumsum(delta_s_position)
    L0 = data.s_position[-1] / 1000
    data.s_position /= data.s_position[-1]
    data.s_director = data.s_position.copy()
    return data, L0


def create_BR2(n=500, L0=0.16, radius=0.0075, youngs_modulus=1e7):
    damp_coefficient = 0.03

    radii = radius * np.ones(n + 1)
    radii_mean = (radii[:-1] + radii[1:]) / 2
    rod = CosseratRod.straight_rod(
        n_elements=n,
        start=np.zeros((3,)),
        direction=np.array([0.0, 1.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        # normal=np.array([0.0, 0.0, 1.0]),
        base_length=L0,
        base_radius=radii_mean.copy(),
        density=700,
        nu=damp_coefficient * ((radii_mean / radius) ** 2),
        youngs_modulus=youngs_modulus,
        poisson_ratio=0.5,
        nu_for_torques=damp_coefficient * ((radii_mean / radius) ** 4),
    )
    return rod


def process_data(file_name, delta_s_position):
    raw_data = read_data_from_file(file_name)
    data, L0 = create_data_object(raw_data, delta_s_position)

    # define rod
    n_elem = 100
    radius = 0.0075 * 2.742
    rod = create_BR2(n=n_elem, L0=L0, radius=radius)

    # define smoothing algorithm
    algo_config = EmptyClass()
    algo_config.argument_weight = 1
    algo_config.step_size = 1e-6
    algo_config.data_deviation_weight_cost_position = 1_000_000
    algo_config.data_deviation_weight_cost_director = 1_000_000

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
        cost = algo.run(iter_number=100_000, threshold=5e-5)
        runtime = time.time() - stime

        # Report delta
        delta_cost = np.abs((cost[-1] - cost[-2]) / cost[-2])
        delta_position = (
            np.abs(smoothed_data["position"][-1] - algo.position).max() * 1000
        )
        delta_director = np.abs(smoothed_data["director"][-1] - algo.director).max()
        delta_shear = np.abs(smoothed_data["shear"][-1] - algo.shear).max()
        delta_kappa = np.abs(smoothed_data["kappa"][-1] - algo.kappa).max()
        num_iter = len(cost)
        print(
            f"time={k} : {runtime=:0.2e} sec, {delta_cost=:0.2e}, {delta_position=:0.2f} mm, {delta_director=:0.2e}, {delta_shear=:0.2e}, {delta_kappa=:0.2e}, {num_iter=}"
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
    import pickle

    print("Saving data to pickle files ...", end="\r")

    with open("result/" + file_name + ".pickle", "wb") as data_file:
        data = dict(
            datafile_name=file_name,
            time=smoothed_data["time"],
            data_index=smoothed_data["data_index"],
            radius=smoothed_data["radius"],
            position=smoothed_data["position"],
            director=smoothed_data["director"],
            shear=smoothed_data["shear"],
            kappa=smoothed_data["kappa"],
        )
        pickle.dump(data, data_file)


def main(problem):
    if problem == "bend":
        file_name = "bend"
        delta_s_position = np.array([23.9, 35.17, 33.82, 34.55, 32.8])
    if problem == "twist":
        file_name = "bend"
        delta_s_position = np.array([23.9, 35.17, 33.82, 34.55, 32.8])
    if problem == "mix":
        file_name = "mix"
        delta_s_position = np.array([23.9, 35.17, 33.82, 34.55, 32.8])
    if problem == "cable":
        file_name = "cable"
        delta_s_position = np.array(
            [27.5, 33.5, 28, 34, 30, 31, 36.5, 31.5, 32, 34.5, 30, 31]
        )

    process_data(file_name, delta_s_position)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Require problem keyword")
    parser.add_argument(
        "--problem",
        metavar="subproblem number",
        type=str,
        nargs=1,
        help="problem keywork: bend, twist, mix, cable",
    )
    args = parser.parse_args()
    main(args.problem[0])
