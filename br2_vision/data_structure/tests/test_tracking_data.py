import h5py
import numpy as np
import pytest

from br2_vision.data_structure import FlowQueue, MarkerPositions, TrackingData


@pytest.fixture
def mock_flow_queues():
    return [
        FlowQueue((0, 0), 0, 0, 0, 0, "A", False),
        FlowQueue((10, 10), 10, 50, 1, 0, "B", True),
        FlowQueue((10, 10), 0, 10, 2, 1, "C", False),
        FlowQueue((10, 10), 0, 10, 3, 2, "D", False),
    ]


def test_initialization_and_cache(mocker, tmp_path, mock_flow_queues):
    cache_path = tmp_path / "cache_tracking_data.h5"
    marker_positions = MarkerPositions([], {})
    spy_mocker_h5_create = mocker.spy(marker_positions, "to_h5")

    assert not cache_path.exists()
    cls = TrackingData.initialize(cache_path, marker_positions)
    spy_file_create = mocker.spy(cls, "create_template")
    with cls:
        for _queue in mock_flow_queues:
            cls.append(_queue)
        assert cls.path == cache_path
        assert cls.marker_positions == marker_positions
        assert cls.queues == mock_flow_queues
        assert cls._inside_context == True
        assert spy_file_create.called
        assert spy_mocker_h5_create.called
    assert not cls._inside_context
    assert cache_path.exists()
    assert cls._inside_context == False

    with h5py.File(cache_path, "r") as h5f:
        assert "marker_positions" in h5f

    with TrackingData.initialize(cache_path, None) as cls:
        assert cls.queues == mock_flow_queues
        # !! this initialize is not a constructor but calls from cache.
        # Therefore, marker_positions is not None.
        assert cls.marker_positions == marker_positions


def test_load_and_match_queue_tags(tmp_path, mock_flow_queues):
    cache_path = tmp_path / "cache_tracking_data.h5"
    marker_positions = MarkerPositions([], {})
    cls = TrackingData.initialize(cache_path, marker_positions)
    with cls:
        for _queue in mock_flow_queues:
            cls.append(_queue)
    for cls_q, mock_q in zip(cls.queues, mock_flow_queues):
        assert cls_q.get_tag() == mock_q.get_tag()


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
        assert cls.iter_cameras() == sorted({1, 2, 3, 0})


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


def test_save_pixel_flow(tmp_path, mock_flow_queues):
    cache_path = tmp_path / "cache_tracking_data.h5"
    marker_positions = MarkerPositions([], {})

    queue = mock_flow_queues[1]
    video_length = 100
    data_length = queue.end_frame - queue.start_frame
    data = np.ones([data_length, 2], dtype=np.int_)

    cls = TrackingData.initialize(cache_path, marker_positions)
    with cls:
        for _queue in mock_flow_queues:
            cls.append(_queue)

    # Test calling function outside of context
    with pytest.raises(Exception):
        cls.save_pixel_flow_trajectory(data, queue, video_length)

    # Test normal save call
    with cls:
        cls.save_pixel_flow_trajectory(data, queue, video_length)

    # Test data
    with h5py.File(cache_path, "r") as h5f:
        grp = h5f.require_group(queue.h5_directory)
        assert grp["xy"].attrs["unit"] == "pixel"
        assert grp["xy"][queue.start_frame : queue.end_frame].shape == data.shape
        np.testing.assert_allclose(grp["xy"][queue.start_frame : queue.end_frame], data)
        np.testing.assert_allclose(grp["xy"][: queue.start_frame], -1)
        np.testing.assert_allclose(grp["xy"][queue.end_frame :], -1)

    # Test re-call
    with cls:
        # Test re-call with different video length
        with pytest.raises(AssertionError):
            cls.save_pixel_flow_trajectory(data, queue, video_length + 1)

        # Test re-call with different data
        cls.save_pixel_flow_trajectory(data + 1, queue, video_length)

    # Test data insertion
    with h5py.File(cache_path, "r") as h5f:
        grp = h5f.require_group(queue.h5_directory)
        np.testing.assert_allclose(
            grp["xy"][queue.start_frame : queue.end_frame], data + 1
        )


def test_load_pixel_flow(tmp_path, mock_flow_queues):
    cache_path = tmp_path / "cache_tracking_data.h5"
    marker_positions = MarkerPositions([], {})

    queue = mock_flow_queues[1]
    video_length = 100
    data_length = queue.end_frame - queue.start_frame
    data = np.ones([data_length, 2], dtype=np.int_)

    cls = TrackingData.initialize(cache_path, marker_positions)
    with cls:
        for _queue in mock_flow_queues:
            cls.append(_queue)
        cls.save_pixel_flow_trajectory(data, queue, video_length)

    # Test calling function outside of context
    with pytest.raises(Exception):
        cls.load_pixel_flow_trajectory(queue, video_length + 1)

    # Test normal load call
    with cls:
        returned_data = cls.load_pixel_flow_trajectory(queue)
    np.testing.assert_allclose(returned_data, data)

    # Test full trajectory
    with cls:
        returned_data = cls.load_pixel_flow_trajectory(queue, full_trajectory=True)
    np.testing.assert_allclose(returned_data[queue.start_frame : queue.end_frame], data)
    np.testing.assert_allclose(returned_data[queue.end_frame :], -1)
    np.testing.assert_allclose(returned_data[: queue.start_frame], -1)


def test_trim_trajectory(tmp_path, mock_flow_queues):
    cache_path = tmp_path / "cache_tracking_data.h5"
    marker_positions = MarkerPositions([], {})

    qid = 1
    queue = mock_flow_queues[qid]
    video_length = 100
    data_length = queue.end_frame - queue.start_frame
    data = np.ones([data_length, 2], dtype=np.int_)

    cls = TrackingData.initialize(cache_path, marker_positions)
    with cls:
        for _queue in mock_flow_queues:
            cls.append(_queue)
        cls.save_pixel_flow_trajectory(data, queue, video_length)

    # Test calling function outside of context
    with pytest.raises(Exception):
        cls.trim_trajectory(queue.get_tag(), 40)

    # Test out-of-bound call
    with cls:
        cls.trim_trajectory(queue.get_tag(), 80)
        returned_data = cls.load_pixel_flow_trajectory(queue)
    np.testing.assert_allclose(returned_data, data)

    # Test normal call
    with cls:
        cls.trim_trajectory(queue.get_tag(), 40)
        returned_data = cls.load_pixel_flow_trajectory(queue)
    np.testing.assert_allclose(returned_data, data[:30])

    # Test reverse call
    with cls:
        cls.trim_trajectory(queue.get_tag(), 20, reverse=True)
        returned_data = cls.load_pixel_flow_trajectory(queue)
    np.testing.assert_allclose(returned_data, data[10:30])

    # Check if queue is updated
    with h5py.File(cache_path, "r") as h5f:
        dset = h5f["queues"][qid]
        q = dset
        np.testing.assert_allclose(q[0], queue.point)
        assert q[1] == 20
        assert q[2] == 40
        assert str(q[5], "utf-8") == "B"


def test_flow_queue_query(tmp_path, mock_flow_queues):
    cache_path = tmp_path / "cache_tracking_data.h5"
    marker_positions = MarkerPositions([], {})

    cls = TrackingData.initialize(cache_path, marker_positions)
    with cls:
        for _queue in mock_flow_queues:
            cls.append(_queue)

    # Test query : by default, query non-done queues
    with cls:
        assert cls.get_flow_queues() == [
            queue for queue in mock_flow_queues if not queue.done
        ]

    # Test query by camera
    with cls:
        assert cls.get_flow_queues(camera=0) == [mock_flow_queues[0]]
        assert cls.get_flow_queues(camera=1) == []
        assert cls.get_flow_queues(camera=1, force_run_all=True) == [
            mock_flow_queues[1]
        ]
        assert cls.get_flow_queues(camera=2) == [mock_flow_queues[2]]
        assert cls.get_flow_queues(camera=3) == [mock_flow_queues[3]]

    # Test query by start_frame
    with cls:
        assert cls.get_flow_queues(start_frame=0) == [
            mock_flow_queues[0],
            mock_flow_queues[2],
            mock_flow_queues[3],
        ]
        assert cls.get_flow_queues(start_frame=10) == []
        assert cls.get_flow_queues(start_frame=10, force_run_all=True) == [
            mock_flow_queues[1]
        ]

    # Test query by tag
    with cls:
        assert cls.get_flow_queues(tag="z0-A") == [mock_flow_queues[0]]
        assert cls.get_flow_queues(tag="z0-B") == []
        assert cls.get_flow_queues(tag="z0-B", force_run_all=True) == [
            mock_flow_queues[1]
        ]
        assert cls.get_flow_queues(tag="z1-C") == [mock_flow_queues[2]]
        assert cls.get_flow_queues(tag="z2-D") == [mock_flow_queues[3]]
