import cv2
import pytest
from click.testing import CliRunner

import br2_vision
from br2_vision.optical_flow import CameraOpticalFlow


def test_run_optical_flow_with_several_points_at_different_timestamps(
    mocker, flowqueue_set_with_different_timestamps, template_br2vision_config
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
    mm = mocker.patch("br2_vision.optical_flow.CameraOpticalFlow.next_inquiry")
    mm.return_value = (mocker.MagicMock(), None)

    # run
    from br2_vision_cli.run_optical_flow import main

    runner = CliRunner()

    result = runner.invoke(
        main,
        ["--tag", "test", "--run-id", "0"],
    )
    assert result.exit_code == 0

    assert mm.call_count == 6
    assert mm.call_args_list == [
        mocker.call([0, 1], 0, 1),
        mocker.call([2], 0, 2),
        mocker.call([3], 1, 2),
        mocker.call([4], 1, 3),
        mocker.call([0], 0, 1),
        mocker.call([0], 0, 1),
    ]
