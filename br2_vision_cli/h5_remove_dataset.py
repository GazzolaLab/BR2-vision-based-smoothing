import os
import sys

import click
import h5py

import br2_vision
from br2_vision.utility.logging import get_script_logger

# ---------------------------- Config ----------------------------
logger = get_script_logger(os.path.basename(__file__))


@click.command()
@click.option(
    "-p",
    "--path",
    type=click.Path(exists=True),
    required=True,
    help="h5 file path",
)
@click.option(
    "-d",
    "--directory",
    type=str,
    required=True,
    help="Dataset to remove",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
def launch(path, directory, verbose):
    with h5py.File(path, "r+") as f:
        if directory in f:
            del f[directory]
            logger.info(f"Dataset '{directory}' removed from '{path}'")
        else:
            logger.warning(f"Dataset '{directory}' not found in '{path}'")
