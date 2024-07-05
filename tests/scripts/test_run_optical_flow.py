import functools
import sys

import click
import cv2
import pytest
from click.testing import CliRunner

import br2_vision
from br2_vision.optical_flow import CameraOpticalFlow


@pytest.fixture
def cli():
    """Yield a click.testing.CliRunner to invoke the CLI."""
    class_ = click.testing.CliRunner

    def invoke_wrapper(f):
        """Augment CliRunner.invoke to emit its output to stdout.

        This enables pytest to show the output in its logs on test
        failures.

        """

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            echo = kwargs.pop("echo", False)
            result = f(*args, **kwargs)

            if echo is True:
                sys.stdout.write(result.output)

            return result

        return wrapper

    class_.invoke = invoke_wrapper(class_.invoke)
    cli_runner = class_()

    yield cli_runner


def test_run_optical_flow_with_several_points_at_different_timestamps(
    mocker, flowqueue_set_with_different_timestamps, template_br2vision_config, cli
):
    mocker.patch("cv2.destroyAllWindows")
    mocker.patch("br2_vision.load_config", return_value=template_br2vision_config)

    # Mock Setup
    def mock_get_flow_queues(camera, force_run_all=False):
        if force_run_all:
            return [
                q for q in flowqueue_set_with_different_timestamps if q.camera == camera
            ]
        else:
            return [
                q
                for q in flowqueue_set_with_different_timestamps
                if q.camera == camera and not q.done
            ]

    _dataset = mocker.MagicMock()
    dataset = mocker.MagicMock()
    dataset.iter_cameras.return_value = {
        q.camera for q in flowqueue_set_with_different_timestamps
    }
    dataset.get_flow_queues = mock_get_flow_queues
    mocker.patch("br2_vision.data_structure.TrackingData.load", return_value=_dataset)
    _dataset.__enter__.return_value = dataset

    mocker.patch(
        "br2_vision.optical_flow.CameraOpticalFlow.num_frames", return_value=10
    )
    mocker.patch("br2_vision.optical_flow.CameraOpticalFlow.render_tracking_video")
    mm = mocker.patch("br2_vision.optical_flow.CameraOpticalFlow")

    def mock_next_inquiry(self, inquiry, stime, etime, debug=False):
        return np.arange(len(inquiry)), None

    mm.next_inquiry = mock_next_inquiry

    # run
    from br2_vision_cli.run_optical_flow import main

    result = cli.invoke(
        main,
        ["--tag", "test", "--run-id", "0"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # assert len(call_args_list) == 6
    # assert call_args_list == [
    #    mocker.call([0, 1], 0, 1),
    #    mocker.call([2], 0, 2),
    #    mocker.call([3], 1, 2),
    #    mocker.call([4], 1, 3),
    #    mocker.call([0], 0, 1),
    #    mocker.call([0], 0, 1),
    # ]
