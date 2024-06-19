import pytest

from br2_vision.data_structure import FlowQueue


def test_flow_queue_change_adjustable_parameters():
    flow_queue = FlowQueue((1, 2), 3, 4, 5, 6, "7")
    flow_queue.start_frame = 8
    flow_queue.end_frame = 9
    flow_queue.done = True

    expected_flow_queue = FlowQueue((1, 2), 8, 9, 5, 6, "7", True)
    assert flow_queue == expected_flow_queue


def test_flow_queue_change_fixed_parameters():
    flow_queue = FlowQueue((1, 2), 3, 4, 5, 6, "7")

    # The point is a fixed parameter, so it should not change.
    with pytest.raises(AttributeError):
        flow_queue.point = (8, 9)
    with pytest.raises(TypeError):  # Point must be a tuple
        flow_queue.point[0] = 8
    with pytest.raises(AttributeError):
        flow_queue.camera = 10
    with pytest.raises(AttributeError):
        flow_queue.z_index = 11
    with pytest.raises(AttributeError):
        flow_queue.label = "5"


def test_flow_queue_typecheck():
    flow_queue = FlowQueue((1, 2), 3, 4, 5, 6, "7")

    # The point is a fixed parameter, so it should not change.
    with pytest.raises(TypeError):
        flow_queue.point = (8, "9")
    with pytest.raises(TypeError):
        flow_queue.point = [8, 9]
    with pytest.raises(TypeError):
        flow_queue.camera = "10"
    with pytest.raises(TypeError):
        flow_queue.z_index = "11"
    with pytest.raises(TypeError):
        flow_queue.label = 5
    with pytest.raises(TypeError):
        flow_queue.done = 5
