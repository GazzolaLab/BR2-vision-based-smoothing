__doc__ = """
Load `br2_vision.ini` file
"""

import os
import pathlib
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path


def load_config(path: "str | Path" = "br2_vision.ini"):
    """
    Load `br2_vision.ini` file
    """
    if isinstance(path, Path):
        path = path.as_posix()
    assert os.path.exists(
        path
    ), f"Configuration file is missing. Please copy the configuration template (template/br2_vision.ini) from the repository."

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path)
    return config
