import pytest

from br2_vision.data_structure import FlowQueue


def test_flowqueue_eq():
    flow_queue = FlowQueue((1, 2), 3, 4, 5, 6, "7")
    flow_queue2 = FlowQueue((1, 2), 3, 4, 5, 6, "7")
    assert flow_queue == flow_queue2


def test_flowqueue_initialized_flag():
    flow_queue = FlowQueue((1, 2), 3, 4, 5, 6, "7")
    assert flow_queue.__initialized__


@pytest.mark.parametrize(
    "z_index, label, expected", [(1, "1", "z1-1"), (2, "2", "z2-2")]
)
def test_flowqueue_tag_str_match(z_index: int, label: str, expected: str):
    flow_queue = FlowQueue((1, 2), 3, 4, 5, z_index, label)
    assert flow_queue.get_tag() == expected


def test_flowqueue_to_numpyarray():
    import numpy as np

    flow_queue = FlowQueue((1, 2), 3, 4, 5, 6, "7")
    np_fq = np.array(flow_queue)
    assert np_fq.dtype == FlowQueue.dtype


def test_flowqueue_h5_directory():
    flow_queue = FlowQueue((1, 2), 3, 4, 5, 6, "700")
    h5_directory = "/trajectory/camera_5/z_6/label_700"
    assert flow_queue.h5_directory == h5_directory
