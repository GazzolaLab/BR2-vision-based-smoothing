import numpy as np
import os, sys

from utility.convert_coordinate import get_center_and_normal

from config import *

import br2_vision
from br2_vision.dlt import DLT
from br2_vision.utility.logging import config_logging, get_script_logger

import click


@click.command()
@click.option(
    "-t",
    "--tag",
    type=str,
    help="Experiment tag. Path ./tag should exist.",
)
@click.option("-r", "--run_id", required=True, type=int, help="Run ID")
def main(tag, runid, n_ring):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    # Read DLT Point Data

    file_path = config["PATHS"]["position_data_path"].format(tag, run_id)
    data = np.load(file_path)

    position_collection = data["position"]
    tags = data["tags"]
    timelength = position_collection.shape[1]

    # Find cross-section data for each timeframe
    cross_section_center_position = []
    cross_section_director = []
    for time in range(timelength):
        labelled_points = {}
        for tag, points in zip(tags, position_collection[:, time, :]):
            if np.all(np.isnan(points)):
                continue
            labelled_points[tag] = points
        center_position, director_vector = get_center_and_normal(
            labelled_points, n_ring=n_ring
        )
        cross_section_center_position.append(center_position)
        cross_section_director.append(director_vector)
    cross_section_center_position = np.array(cross_section_center_position)
    cross_section_director = np.array(cross_section_director)

    # Append in the same file
    print(f"{cross_section_center_position.shape=}")
    print(f"{cross_section_director.shape=}")
    print(cross_section_center_position[0, ...])
    data = dict(data)
    data["cross_section_center_position"] = cross_section_center_position
    data["cross_section_director"] = cross_section_director
    np.savez(
        file_path,
        **data,
    )

    # Verbose
    print("Data saved: {}".format(output_file_path))


if __name__ == "__main__":
    runid = 1
    main(runid, n_ring=5)
