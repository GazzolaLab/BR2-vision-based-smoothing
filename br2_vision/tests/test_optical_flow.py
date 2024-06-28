import pytest

from br2_vision.data_structure import FlowQueue, TrackingData
from br2_vision.optical_flow import CameraOpticalFlow


# Make flow queue fixtures
@pytest.fixture
def mock_flow_queues():
    return [
        FlowQueue((0, 0), 0, 0, 0, 0, "A", False),
        FlowQueue((10, 10), 0, 0, 0, 0, "B", True),
        FlowQueue((10, 10), 0, 10, 0, 0, "C", False),
        FlowQueue((10, 10), 0, 10, 0, 0, "D", False),
    ]


class MockTrackingData:
    data_history = []
    q_history = []
    size_history = []

    def save_pixel_flow_trajectory(self, data, q, size):
        self.data_history.append(data)
        self.q_history.append(q)
        self.size_history.append(size)


def test_make_OF_module(tmp_path, mock_flow_queues):
    tracking_data = TrackingData(tmp_path, None)
    of = CameraOpticalFlow(tmp_path, mock_flow_queues, tracking_data)
    assert of is not None


def test_num_frames(mocker):
    mocker.patch(
        "br2_vision.cv2_custom.extract_info.get_video_frame_count", return_value=0
    )
    of = CameraOpticalFlow(None, [], None)
    assert of.num_frames == 0

    # User cannot set num_frames
    of.___num_frames = 10
    assert of.num_frames != 10
    assert of.num_frames == 0


def test_run_optical_flow(mocker, mock_flow_queues):
    mocker.patch(
        "br2_vision.cv2_custom.extract_info.get_video_frame_count", return_value=0
    )
    mocker.patch("br2_vision.optical_flow.CameraOpticalFlow.num_frames", return_value=10)

    tracking_data = MockTrackingData()
    of = CameraOpticalFlow(None, mock_flow_queues, tracking_data)
    of.next_inquiry = lambda x, y, z: (range(len(x)), None)
        #of, "next_inquiry", return_value=(list(range(len([q for q in mock_flow_queues if not q.done]))), None)
    #)
    # Test debug mode
    dones = [q.done for q in mock_flow_queues]
    of.run(debug=True)
    assert [q.done for q in mock_flow_queues] == dones

    of.run()
    assert all([q.done for q in mock_flow_queues])

    assert tracking_data.data_history == [0, 0, 1]  # [0] and [0,1]
    assert tracking_data.q_history == [mock_flow_queues[0], mock_flow_queues[2], mock_flow_queues[3]]
