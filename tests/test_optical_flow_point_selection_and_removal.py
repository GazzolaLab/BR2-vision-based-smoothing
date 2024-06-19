import os
import pathlib
import sys

import numpy as np
import pytest

import br2_vision
from br2_vision.data_structure import FlowQueue, MarkerPositions, TrackingData

# Create similar action sequence as add_initial_flow_point


def test_access_br2_vision_ini(template_br2vision_config):
    config = template_br2vision_config
    assert True


def test_add_optical_flow_point(template_br2vision_config):
    config = template_br2vision_config
    scale = float(config["DIMENSION"]["scale_video"])

    # Arbitrary values
    tag = "exp"
    cam_id = 0
    run_id = [0]
    start_frame = 50
    end_frame = 100
    video_length = 350
    assert start_frame < video_length

    # Path
    marker_positions = MarkerPositions.from_yaml(config["PATHS"]["marker_positions"])
    initial_point_file = config["PATHS"]["tracing_data_path"].format(tag, run_id[0])
    keys = marker_positions.tags

    with TrackingData.initialize(
        path=initial_point_file, marker_positions=marker_positions
    ) as dataset:
        flow_queues = dataset.get_flow_queues(
            camera=cam_id, start_frame=start_frame, force_run_all=True
        )
        old_points = [queue.point for queue in flow_queues]  # (N, 2)
        old_marker_label = [
            (queue.z_index, queue.label) for queue in flow_queues
        ]  # (N, 2)

    marker_label = [("1", "a"), ("2", "b"), ("3", "c")]
    points = [[100, 200], [200, 300], [300, 400]]

    # Load existing points and marker_label, and append
    # TODO: Maybe use loaded queue from the beginning
    for rid in run_id:

        initial_point_file = config["PATHS"]["tracing_data_path"].format(tag, rid)
        with TrackingData.initialize(
            path=initial_point_file,
            marker_positions=marker_positions,
        ) as dataset:
            # print(f"{len(dataset.get_flow_queues(camera=cam_id))=}")
            for label, point in zip(marker_label, points):
                point = tuple([int(v) for v in point])
                flow_queue = FlowQueue(
                    point, start_frame, end_frame, cam_id, int(label[0]), label[1]
                )
                dataset.append(flow_queue)
                dataset.save_pixel_flow_trajectory(
                    np.ones((end_frame - start_frame, 2), dtype=np.int32),
                    flow_queue,
                    video_length,
                )
                flow_queue.done = True
        assert os.path.exists(initial_point_file)


@pytest.mark.dependency(depends=["test_add_optical_flow_point"])
def test_remove_optical_flow_point(template_br2vision_config):
    config = template_br2vision_config
    scale = float(config["DIMENSION"]["scale_video"])

    # Arbitrary values
    tag = "exp"
    cam_id = 0
    run_id = [0]
    start_frame = 50
    cut_frame = 60
    video_length = 350

    # Path
    marker_positions = MarkerPositions.from_yaml(config["PATHS"]["marker_positions"])
    initial_point_file = config["PATHS"]["tracing_data_path"].format(tag, run_id[0])

    with TrackingData.initialize(
        path=initial_point_file, marker_positions=marker_positions
    ) as dataset:
        dataset.trim_trajectory("z1-a", cut_frame)
        flowqueue = dataset.get_flow_queues(
            camera=cam_id, tag="z1-a", force_run_all=True
        )[0]
        assert flowqueue.end_frame == cut_frame
        assert flowqueue.start_frame == start_frame
        assert dataset.load_pixel_flow_trajectory(flowqueue).shape == (
            cut_frame - start_frame,
            2,
        )
        assert dataset.load_pixel_flow_trajectory(
            flowqueue, full_trajectory=True
        ).shape == (video_length, 2)


@pytest.mark.dependency(depends=["test_remove_optical_flow_point"])
def test_re_add_optical_flow_point(template_br2vision_config):
    config = template_br2vision_config

    # Arbitrary values
    tag = "exp"
    cam_id = 0
    run_id = [0]
    start_frame = 61
    end_frame = 75
    video_length = 350

    # New point
    marker_label = [("4", "d"), ("1", "a")]
    points = [[100, 200], [100, 300]]

    marker_positions = MarkerPositions.from_yaml(config["PATHS"]["marker_positions"])
    initial_point_file = config["PATHS"]["tracing_data_path"].format(tag, run_id[0])
    with TrackingData.initialize(
        path=initial_point_file, marker_positions=marker_positions
    ) as dataset:
        flow_queues = dataset.get_flow_queues(camera=cam_id, force_run_all=True)
        old_points = [queue.point for queue in flow_queues]  # (N, 2)
        old_marker_label = [
            (queue.z_index, queue.label) for queue in flow_queues
        ]  # (N, 2)
        for label, point in zip(marker_label, points):
            point = tuple([int(v) for v in point])
            flow_queue = FlowQueue(
                point, start_frame, end_frame, cam_id, int(label[0]), label[1]
            )
            dataset.append(flow_queue)
            dataset.save_pixel_flow_trajectory(
                np.ones((end_frame - start_frame, 2), dtype=np.int32),
                flow_queue,
                video_length,
            )

        flowqueues = dataset.get_flow_queues(camera=cam_id, tag="z1-a")
        flowqueue = flowqueues[0]
        assert flowqueue.end_frame == end_frame
        assert flowqueue.start_frame == start_frame
        assert dataset.load_pixel_flow_trajectory(flowqueue).shape == (
            end_frame - start_frame,
            2,
        )
        assert dataset.load_pixel_flow_trajectory(
            flowqueue, full_trajectory=True
        ).shape == (video_length, 2)
