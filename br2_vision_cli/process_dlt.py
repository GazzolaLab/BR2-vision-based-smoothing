import os
import sys
from pathlib import Path
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
from br2_vision.data_structure.marker_positions import MarkerPositions
from br2_vision.data_structure.posture_data import PostureData
from br2_vision.data_structure.tracking_data import TrackingData
from br2_vision.dlt import DLT
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
@click.option(
    "-p",
    "--save_path",
    type=click.Path(exists=False),
    default="results",
    help="Path to save the data",
)
@click.option("-v", "--verbose", is_flag=True, type=bool, help="Verbose output")
def process_dlt(tag, run_id, fps, save_path, verbose):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    tracing_data_path = config["PATHS"]["tracing_data_path"].format(tag, run_id)
    assert os.path.exists(
        tracing_data_path
    ), f"Tracing data does not exist: path={tracing_data_path}"
    with TrackingData.load(path=tracing_data_path) as dataset:
        # FIXME: redundant load
        marker_positions = MarkerPositions.from_yaml(
            config["PATHS"]["marker_positions"]
        )
        dataset.marker_positions = marker_positions

        n_ring = len(marker_positions)
        ring_space = marker_positions.marker_center_offset

        # Read Data Points
        timelength = dataset.query_timelength()
        timestamps = np.arange(0, timelength) * (1.0 / fps)
        dataset.save_timestamps(timestamps)

        # Count observed tags for each camera
        tags_count = defaultdict(int)
        recorded_labels = {
            (q.camera, q.z_index, q.label) for q in dataset.queues if q.done
        }
        for cid, zid, label in recorded_labels:
            tags_count[(zid, label)] += 1

        # Remove tags if count is less than 2
        observing_camera_count = {k: v for k, v in tags_count.items() if v >= 2}

        # Integrity check: Print if 3 labels exists for each camera for all frames
        flag = True
        for zid in range(n_ring):

            count_label = {
                label: count
                for (z, label), count in observing_camera_count.items()
                if z == zid
            }
            num_labels_at_zid = len(count_label)
            if num_labels_at_zid < 3:
                flag = False
                print(f"Z={zid}: False - {num_labels_at_zid} labels traced.")
            else:
                print(f"Z={zid}: True")
                for label, count in count_label.items():
                    print(f"  {label}: {count} times.")
        assert flag, "At least three labels should be interpolated for each z-plane."
        result_tags = list(observing_camera_count.keys())

        # Verbose output
        print("Total number of observed tags: ", len(observing_camera_count))
        print("Used Tags: ", result_tags)

        # DLT Interpolation
        dlt = DLT(calibration_path=config["PATHS"]["calibration_path"])
        dlt.load()
        result_dlt_coords = []
        result_actual_coords = []
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
                    uvs[cam_id] = p[tid]
                if len(uvs) < 2:
                    _xyz, cond = np.zeros(3) * np.nan, np.inf
                else:
                    _xyz, cond = dlt.map(uvs)
                txyz.append(_xyz)
                conds.append(cond)

            if not np.isnan(txyz[0]).any():
                actual_coord = marker_positions.get_position(zid, label)
                result_dlt_coords.append(txyz[0])  # keep the firat frame
                result_actual_coords.append(actual_coord)

            dataset.save_track(np.asarray(txyz), zid, label, prefix="dlt_mapped_xyz")

        result_dlt_coords = np.array(result_dlt_coords)
        result_actual_coords = np.array(result_actual_coords)
        print("Process tags:")
        print("Total number of processed timesteps: {}".format(timelength))
        print(
            "Total number of processed points: {}".format(len(observing_camera_count))
        )

        # Re-map dlt points to rod space
        reg = LinearRegression(fit_intercept=True).fit(
            result_dlt_coords, result_actual_coords
        )
        print("regression coefficient:\n", reg.coef_)
        print("   det=", np.linalg.det(reg.coef_))
        print("   trace=", np.trace(reg.coef_))
        print("   angle=", np.arccos((np.trace(reg.coef_) - 1) / 2.0))
        print("regression rank: ", reg.rank_)
        print("regression intercept: ", reg.intercept_)
        print("regression score: ", reg.score(result_dlt_coords, result_actual_coords))

        for tag, count in observing_camera_count.items():
            zid, label = tag
            txyz = dataset.load_track(zid, label, prefix="dlt_mapped_xyz")
            nan_indices = np.isnan(txyz).any(axis=1)
            mapped_txyz = reg.predict(txyz[~nan_indices])
            txyz[~nan_indices] = mapped_txyz
            dataset.save_track(txyz, zid, label, prefix="xyz")

    with PostureData(path=tracing_data_path) as dataset:
        pass

    # TODO: debugging plots
    # plot_path_activity = config["PATHS"]["plot_working_box"].format(tag, run_id)
    # fig = plt.figure(1, figsize=(10, 8))
    # ax = plt.axes(projection="3d")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # for i in range(len(result_tags)):
    #     # print(result_tags[i], ': ', initial_dlt_cond[i])
    #     ax.scatter(
    #         *result_dlt_coords [i],
    #         color=color_scheme[observing_camera_count[result_tags[i]]],
    #     )
    #     # c = int(initial_dlt_cond[i]*100/initial_dlt_cond_max)-1
    #     # ax.scatter(*result_dlt_coords [i], c=c, cmap='viridis')
    #     # ax.scatter(*result_actual_coords [i])
    #     ax.text(*result_dlt_coords [i], result_tags[i], size=10, zorder=1, color="k")
    # # draw cube
    # cx = np.array([1, 15]) * 0.02
    # cy = np.array([1, 12]) * 0.04
    # cz = np.array([1, 10]) * 0.04
    # for s, e in combinations(np.array(list(product(cx, cy, cz))), 2):
    #     if np.sum(np.abs(s - e)) >= 0.2:
    #         ax.plot3D(*zip(s, e), color="b")
    # ax.view_init(elev=-90, azim=-70)
    # fig.savefig(plot_path_activity)
    # # plt.show()

    # return


if __name__ == "__main__":
    process_dlt()
