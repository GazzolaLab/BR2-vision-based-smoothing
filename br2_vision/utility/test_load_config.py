import pytest

from br2_vision.utility.load_config import load_config

@pytest.fixture
def mock_script():
    return """[PATHS]
# DATA PATH
data_dir: data

# VIDEO DATA PATH
raw_video_path                   : ${data_dir}/raw
    """


def test_load_config(mock_script, tmp_path):
    with open(tmp_path / 'config.ini', 'w') as f:
        f.write(mock_script)

    config = load_config(tmp_path / 'config.ini')

    assert config['PATHS']['data_dir'] == 'data'
    assert config['PATHS']['raw_video_path'] == 'data/raw'

