__doc__ = """
Load `br2_vision.ini` file
"""

import os
from configparser import ConfigParser, ExtendedInterpolation
import pathlib


def load_config(path="br2_vision.ini"):
    """
    Load `br2_vision.ini` file
    """
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path)
    return config
