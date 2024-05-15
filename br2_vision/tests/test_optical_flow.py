import pytest

from br2_vision.optical_flow import CameraOpticalFlow
from br2_vision.data_structure import FlowQueue, TrackingData

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
    mocker.patch("br2_vision.cv2_custom.extract_info.get_video_frame_count", return_value=0)
    of = CameraOpticalFlow(None, [], None)
    assert of.num_frames == 0
    assert of._num_frames == 0

    of._num_frames = 10
    assert of.num_frames == 10

def test_run_optical_flow(mocker, mock_flow_queues):
    mocker.patch("br2_vision.cv2_custom.extract_info.get_video_frame_count", return_value=0)

    tracking_data = MockTrackingData()
    of = CameraOpticalFlow(None, mock_flow_queues, tracking_data)
    mocker.patch.object(of, "next_inquiry", return_value=(list(range(len(mock_flow_queues))), None))
    # Test debug mode
    dones = [q.done for q in mock_flow_queues]
    of.run(debug=True)
    assert [q.done for q in mock_flow_queues] == dones

    of.run()
    assert all([q.done for q in mock_flow_queues])

    assert tracking_data.data_history == [0, 2, 3]  # queue already happened
