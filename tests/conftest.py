import os
import pathlib

import pytest

import br2_vision


@pytest.fixture(scope="session")
def template_br2vision_config(tmp_path_factory):
    # Within pytest: access "../template/br2_vision.ini"
    config_path = (
        pathlib.Path(os.path.dirname(__file__)).resolve().parent
        / "template"
        / "br2_vision.ini"
    )
    marker_positions_path = config_path.parent / "markers.yaml"

    # Create a temporary directory
    tmp_path = tmp_path_factory.mktemp("test_dir")

    config = br2_vision.load_config(config_path)
    config["PATHS"]["data_dir"] = tmp_path.as_posix()
    config["PATHS"]["marker_positions"] = marker_positions_path.as_posix()

    assert os.path.exists(config["PATHS"]["data_dir"])
    assert os.path.exists(config["PATHS"]["marker_positions"])

    return config
