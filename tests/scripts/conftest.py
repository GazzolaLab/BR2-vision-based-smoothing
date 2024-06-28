import pytest

from br2_vision.data_structure import FlowQueue


@pytest.fixture(scope="session")
def flowqueue_set_with_different_timestamps():
    # Mixed camera, start_frame, and done
    return [
        # Camera 0
        FlowQueue(
            point=(0, 0),
            start_frame=0,
            end_frame=1,
            camera=0,
            z_index=1,
            label="R",
            done=False,
        ),
        FlowQueue(
            point=(0, 0),
            start_frame=0,
            end_frame=1,
            camera=0,
            z_index=1,
            label="R2",
            done=False,
        ),
        FlowQueue(
            point=(0, 0),
            start_frame=0,
            end_frame=2,
            camera=0,
            z_index=1,
            label="R",
            done=False,
        ),
        FlowQueue(
            point=(0, 0),
            start_frame=1,
            end_frame=2,
            camera=0,
            z_index=1,
            label="R",
            done=False,
        ),
        FlowQueue(
            point=(0, 0),
            start_frame=1,
            end_frame=3,
            camera=0,
            z_index=1,
            label="R",
            done=False,
        ),
        # Camera 1
        FlowQueue(
            point=(0, 0),
            start_frame=0,
            end_frame=1,
            camera=1,
            z_index=5,
            label="R",
            done=False,
        ),
        # Camera 2
        FlowQueue(
            point=(0, 0),
            start_frame=0,
            end_frame=1,
            camera=2,
            z_index=4,
            label="R",
            done=False,
        ),
        # Done
        FlowQueue(
            point=(0, 0),
            start_frame=0,
            end_frame=1,
            camera=0,
            z_index=2,
            label="R",
            done=True,
        ),
        FlowQueue(
            point=(0, 0),
            start_frame=0,
            end_frame=1,
            camera=3,
            z_index=3,
            label="R",
            done=True,
        ),
    ]
