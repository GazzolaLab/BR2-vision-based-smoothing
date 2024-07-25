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
    type=str,
    required=True,
    help="Dataset to remove",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def launch(path, dataset, verbose, dry):
    with h5py.File(path, "r+") as f:
        if dataset in f:
            if not dry:
                del f[dataset]
                logger.info(f"Dataset '{dataset}' removed from '{path}'")
            else:
                logger.info(f"Dataset '{dataset}' would be removed from '{path}'")
        else:
            logger.warning(f"Dataset '{dataset}' not found in '{path}'")
