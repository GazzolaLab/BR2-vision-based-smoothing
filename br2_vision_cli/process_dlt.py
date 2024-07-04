import os
import sys
from collections import defaultdict
from itertools import combinations, product

import click
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


import br2_vision
from br2_vision.data_structure.tracking_data import TrackingData
from br2_vision.dlt import DLT
from br2_vision.utility.convert_coordinate import three_ring_xyz_converter
from br2_vision.utility.logging import config_logging, get_script_logger

# CONFIGURATION
color_scheme = plt.rcParams["axes.prop_cycle"].by_key()["color"]


@click.command()
@click.option(
    "-t",
    "--tag",
    type=str,
    help="Experiment tag. Path ./tag should exist.",
)
@click.option("-r", "--run_id", required=True, type=int, help="Run ID")
@click.option("-f", "--fps", default=60, type=int, help="FPS of the video")
@click.option("-v", "--verbose", is_flag=True, type=bool, help="Verbose output")
def process_dlt(tag, run_id, fps, verbose):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    tracing_data_path = config["PATHS"]["tracing_data_path"].format(tag, run_id)
    assert os.path.exists(tracing_data_path), f"Tracing data does not exist: path={tracing_data_path}"
    with TrackingData.load(path=tracing_data_path) as dataset:
        marker_positions = dataset.marker_positions

        n_ring = len(marker_positions)
        ring_space = marker_positions.marker_center_offset

        # Read Data Points
        timelength = dataset.query_timelength()
        timestamps = np.arange(0, timelength) * (1.0 / fps)

        # Count observed tags for each camera
        tags_count = defaultdict(int)
        recorded_labels = {(q.camera, q.z_index, q.label) for q in dataset.queues if q.done}
        for cid, zid, label in recorded_labels:
            tags_count[(zid, label)] += 1
        
        # Remove tags if count is less than 2
        observing_camera_count = { k : v for k, v in tags_count.items() if v >= 2 } 

        # Print if 3 labels exists for each camera
        flag = True
        for zid in range(n_ring):
            num_labels_at_zid = len({label for (z, label) in observing_camera_count.keys() if z == zid})
            if num_labels_at_zid != 3:
                flag = False
                print(f"Z={zid}: False - {num_labels_at_zid} labels traced.")
            else:
                print(f"Z={zid}: True")
        assert flag, "At least three labels should be interpolated for each z-plane."
        result_tags = list(observing_camera_count.keys())

        # Verbose output
        print("Total number of observed tags: ", len(observing_camera_count))
        print("Used Tags: ", result_tags)

        # DLT Interpolation
        dlt = DLT(calibration_path=config["PATHS"]["calibration_path"])
        dlt.load()
        points, tags = [], []

        result_points = []
        result_cond = []
        for tag, count in observing_camera_count.items():
            # if tag in EXCLUDE_TAGS:
            #     continue
            zid, label = tag
            point_collections_for_tag = dataset.query_trajectory(zid, label, timelength)
            txyz = []
            conds = []
            for tid in range(timelength):
                uvs = {}
                for cam_id, p in point_collections_for_tag.items():
                    if p[tid, 0] == -1 or p[tid, 1] == -1:
                        continue
                    uvs[cam_id+1] = p[tid]
                _xyz, cond = dlt.map(uvs)
                txyz.append(_xyz)
                conds.append(cond)
            result_points.append(txyz)
            result_cond.append(conds)
        result_points = np.array(result_points)
        result_cond = np.array(result_cond)
    print("Process tags:")
    print("Total number of processed timesteps: {}".format(timelength))
    print("Total number of processed points: {}".format(result_points.shape[1]))
    print("Final result shape: {}".format(result_points.shape))

    # Move to simulation space
    initial_dlt_space = result_points[:, 0, :]
    initial_dlt_cond = result_cond[:, 0]
    initial_dlt_cond_max = initial_dlt_cond.max()
    initial_sim_space = np.empty_like(initial_dlt_space)
    for idx, tag in enumerate(result_tags):
        initial_sim_space[idx, :] = three_ring_xyz_converter(
            tag, n_ring, kwargs.get("ring_space")
        )
    reg = LinearRegression(fit_intercept=True, normalize=False).fit(
        initial_dlt_space, initial_sim_space
    )
    print("regression coefficient:\n", reg.coef_)
    print("   det=", np.linalg.det(reg.coef_))
    print("   trace=", np.trace(reg.coef_))
    print("   angle=", np.arccos((np.trace(reg.coef_) - 1) / 2.0))
    print("regression rank: ", reg.rank_)
    print("regression intercept: ", reg.intercept_)
    print("regression score: ", reg.score(initial_dlt_space, initial_sim_space))
    simulation_space_points = reg.predict(result_points.reshape([-1, 3]))
    simulation_space_points = simulation_space_points.reshape([-1, timelength, 3])

    # result_points = simulation_space_points

    # Output Path - Default path is the data directory
    # TODO: move data into tracing_data_path
    output_points_path = config["PATHS"]["position_data_path"].format(tag, run_id)
    # Save Points
    np.savez(
        output_points_path,
        time=timestamps,
        position=result_points,
        # simulation_space_points=simulation_space_points,
        tags=result_tags,
        # origin_offset=center,
        # normal_pob=np.array(plane.normal)
    )
    print("Points saved at - {}".format(output_points_path))
    print("")

    return 

    plot_path_activity = config["PATHS"]["plot_working_box"].format(tag, run_id)
    fig = plt.figure(1, figsize=(10, 8))
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    for i in range(len(result_tags)):
        # print(result_tags[i], ': ', initial_dlt_cond[i])
        ax.scatter(*initial_dlt_space[i], color=color_scheme[observing_camera_count[result_tags[i]]])
        # c = int(initial_dlt_cond[i]*100/initial_dlt_cond_max)-1
        # ax.scatter(*initial_dlt_space[i], c=c, cmap='viridis')
        # ax.scatter(*initial_sim_space[i])
        ax.text(*initial_dlt_space[i], result_tags[i], size=10, zorder=1, color="k")
    # draw cube
    cx = np.array([1, 15]) * 0.02
    cy = np.array([1, 12]) * 0.04
    cz = np.array([1, 10]) * 0.04
    for s, e in combinations(np.array(list(product(cx, cy, cz))), 2):
        if np.sum(np.abs(s - e)) >= 0.2:
            ax.plot3D(*zip(s, e), color="b")
    ax.view_init(elev=-90, azim=-70)
    fig.savefig(plot_path_activity)
    # plt.show()

    return


if __name__ == "__main__":
    process_dlt()
