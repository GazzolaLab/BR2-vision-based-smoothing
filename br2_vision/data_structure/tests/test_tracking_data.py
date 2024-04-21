import pytest

import h5py

from br2_vision.data_structure import TrackingData, FlowQueue, MarkerPositions


@pytest.fixture
def mock_flow_queues():
    return [
        FlowQueue((0, 0), 0, 0, 0, 0, "A", False),
        FlowQueue((10, 10), 0, 0, 1, 0, "B", True),
        FlowQueue((10, 10), 0, 10, 2, 0, "C", False),
        FlowQueue((10, 10), 0, 10, 3, 0, "D", False),
    ]

def test_initialization_and_cache(mocker, tmp_path, mock_flow_queues):
    cache_path = tmp_path / "cache_tracking_data.h5"
    marker_positions = MarkerPositions([], {})
    spy_mocker_h5_create = mocker.spy(marker_positions, "to_h5")

    assert not cache_path.exists()
    cls = TrackingData.initialize(cache_path, marker_positions)
    spy_file_create = mocker.spy(cls, "create_template")
    with cls:
        assert cls.path == cache_path
        assert cls.marker_positions == marker_positions
        assert cls.queues == []
        assert cls._inside_context
        assert spy_file_create.called
        assert spy_mocker_h5_create.called
    assert not cls._inside_context
    assert cache_path.exists()
    
    with h5py.File(cache_path, "r") as h5f:
        assert "marker_positions" in h5f

    with TrackingData.initialize(cache_path, None) as cls:
        assert cls.queues == []
        # !! this initialize is not a constructor but calls from cache.
        # Therefore, marker_positions is not None.
        assert cls.marker_positions == marker_positions 

def test_done_checker(mocker, tmp_path, mock_flow_queues):
    cache_path = tmp_path / "cache_tracking_data.h5"
    marker_positions = MarkerPositions([], {})
    mocker.patch.object(marker_positions, "to_h5", return_value=None)

    with TrackingData.initialize(cache_path, marker_positions) as cls:
        assert cls.all_done
        cls.queues = mock_flow_queues
        assert not cls.all_done, f"{cls}, {cls.all_done}"
        mock_flow_queues[0].done = True
        mock_flow_queues[2].done = True
        mock_flow_queues[3].done = True
        assert cls.all_done

def test_iter_cameras(tmp_path, mock_flow_queues):
    cache_path = tmp_path / "cache_tracking_data.h5"
    marker_positions = MarkerPositions([], {})

    with TrackingData.initialize(cache_path, marker_positions) as cls:
        for queue in mock_flow_queues:
            cls.queues.append(queue)
        assert cls.iter_cameras() == sorted({1,2,3,0})

def test_append_queue(tmp_path, mock_flow_queues):
    cache_path = tmp_path / "cache_tracking_data.h5"
    marker_positions = MarkerPositions([], {})

    with TrackingData.initialize(cache_path, marker_positions) as cls:
        for queue in mock_flow_queues:
            cls.append(queue)
        assert cls.queues == mock_flow_queues
        # !!! If the queue is already in the list, it should be replaced
        for queue in mock_flow_queues:
            cls.append(queue)
        assert cls.queues == mock_flow_queues
