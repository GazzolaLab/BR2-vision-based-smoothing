import glob
import os
import pathlib
import re
import sys

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

import br2_vision
from br2_vision.cv2_custom.transformation import scale_image
from br2_vision.data_structure import OriginData
from br2_vision.utility.logging import config_logging, get_script_logger


def mouse_event_click_point(event, x, y, flags, param):
    point = param["point"]
    if event == cv2.EVENT_LBUTTONDOWN:
        point[0] = x
        point[1] = y
    param["display_func"]()


@click.command()
@click.option(
    "-t",
    "--tag",
    type=str,
    help="Experiment tag. Path ./tag should exist.",
)
@click.option(
    "-r",
    "--run-id",
    type=int,
    help="Specify run index. Initial points are saved for all specified run-ids.",
    default=0,
)
@click.option(
    "--glob-run-id",
    type=bool,
    default=True,
    is_flag=True,
    help="If specified, use all run-ids in the directory. By default, True.",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def main(tag, run_id, glob_run_id, verbose, dry):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))
    scale = float(config["DIMENSION"]["scale_video"])

    # Path
    video_path = config["PATHS"]["footage_video_path"].format(tag, "*", run_id)

    cam_id = []
    candidates = pathlib.Path(".").glob(
        config["PATHS"]["footage_video_path"].format(tag, "*", run_id)
    )
    pattern = config["PATHS"]["footage_video_path"].format(tag, "(\d+)", run_id)
    for candidate in candidates:
        rr = re.search(pattern, candidate.as_posix())
        if rr is None:
            continue
        rid = int(rr.group(1))
        cam_id.append(rid)

    points = []
    for cid in cam_id:
        video_path = config["PATHS"]["footage_video_path"].format(tag, cid, run_id)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        assert ret
        frame = scale_image(frame, scale)
        cap.release()

        # Select point using mouse event
        point: list[float, float] = [0, 0]
        window_name = "select point"

        def display():
            disp_img = frame.copy()
            cv2.circle(disp_img, tuple(point), 3, (0, 255, 0), -1)
            cv2.imshow(window_name, disp_img)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(
            window_name,
            mouse_event_click_point,
            param={"display_func": display, "point": point},
        )
        display()
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        assert len(point) > 0, "No point selected"
        points.append(point)
    points = np.array(points)

    if glob_run_id:
        candidates = pathlib.Path(".").glob(
            config["PATHS"]["footage_video_path"].format(tag, cam_id[0], "*")
        )
        pattern = config["PATHS"]["footage_video_path"].format(tag, cam_id[0], "(\d+)")
        all_run_id = []
        for candidate in candidates:
            rr = re.search(pattern, candidate.as_posix())
            if rr is None:
                continue
            rid = int(rr.group(1))
            all_run_id.append(rid)

    # Load existing points and marker_label, and append
    print(f"{cam_id=}")
    print(f"{points=}")
    for rid in all_run_id:
        print(f"save {rid=}: ", end="")
        initial_point_file = config["PATHS"]["tracing_data_path"].format(tag, rid)
        with OriginData(path=initial_point_file) as dataset:
            dataset.set_camera_frames(cam_id, points)
        print("done")


if __name__ == "__main__":
    main()
