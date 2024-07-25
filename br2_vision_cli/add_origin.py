import glob
import os
import pathlib
import re
import sys

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import QApplication, QInputDialog, QLineEdit, QPushButton, QWidget

import br2_vision
from br2_vision.cv2_custom.marking import cv2_draw_label
from br2_vision.cv2_custom.transformation import scale_image
from br2_vision.data_structure import OriginData
from br2_vision.qt_custom.label_prompt import LabelPrompt
from br2_vision.utility.logging import config_logging, get_script_logger


def on_mouse_zoom(event, x, y, flags, param):
    uv = param["uv"]
    original_uv = param["original_uv"]
    if event == cv2.EVENT_LBUTTONDOWN:
        uv[0] = x
        uv[1] = y
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Return original uv
        uv[:] = original_uv


def zoomed_inquiry(current_frame, uv, scale=5.0, disp_h=80, disp_w=80):
    """
    Inquiry for a point in a zoomed-in region of the image.
    """
    x, y = uv
    x = int(x)
    y = int(y)

    # Region of interest display
    window_name_roi = "zoom-prompt"
    cv2.namedWindow(window_name_roi)
    disp_img_roi = current_frame.copy()
    disp_img_roi = cv2.rectangle(
        disp_img_roi,
        (x - disp_w // 2, y - disp_h // 2),
        (x + disp_w // 2, y + disp_h // 2),
        (0, 0, 255),
        thickness=3,
    )
    cv2.imshow(window_name_roi, disp_img_roi)

    # Transformation
    img = current_frame.copy()
    padded_img = cv2.copyMakeBorder(
        img,
        disp_h // 2,
        disp_h // 2,
        disp_w // 2,
        disp_w // 2,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    scaled_img = scale_image(padded_img[y : y + disp_h, x : x + disp_w], scale)
    _x = int(disp_w * scale / 2)
    _y = int(disp_h * scale / 2)
    _uv = np.array([_x, _y])

    # Implement mouse event for clicking other point
    original_uv = _uv.copy()

    # Inquiry Loop
    inquiry_on = True
    window_name = "select reappeared point"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name, on_mouse_zoom, param={"uv": _uv, "original_uv": original_uv}
    )
    print("zoomed-in inquiry")
    print("d: cancel, a: accept, h: help")
    while inquiry_on:
        disp_img = scaled_img.copy()

        # Draw cross with exact center _uv
        disp_img[_uv[1] : _uv[1] + 1, :] = np.array([0, 0, 235])
        disp_img[:, _uv[0] : _uv[0] + 1] = np.array([0, 0, 235])

        cv2.imshow(window_name, disp_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):  # Cancel: No point found
            inquiry_on = False
            uv = original_uv
        elif key == ord("a"):  # Accept: accept change
            inquiry_on = False
        elif key == ord("h"):  # Help
            print("d: cancel, a: accept, h: help")
        else:
            pass

    cv2.destroyWindow(window_name)
    cv2.destroyWindow(window_name_roi)

    x = int(_uv[0] / scale) + x - disp_w // 2
    y = int(_uv[1] / scale) + y - disp_h // 2

    return np.array([x, y], dtype=np.int_)


def mouse_event_click_point(event, x, y, flags, param):
    global prev_tag
    points = param["points"]
    marker_label = param["marker_label"]
    bypass_inquiry = flags & cv2.EVENT_FLAG_CTRLKEY
    if event == cv2.EVENT_LBUTTONDOWN:
        point = np.array([x, y], dtype=np.int_).tolist()
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Second zoom-layer selection
        uv = zoomed_inquiry(param["frame"], np.array([x, y]))
        point = uv.tolist()
    else:
        return

    # Ask for a tag in a separate window
    if bypass_inquiry:
        tag = prev_tag
    else:
        tag = param["prompt"]()
        if tag is None:
            print("canceled")
            return
    prev_tag = tag

    points.append(point)
    marker_label.append(tag)
    print("added: ")
    print(point, tag)

    param["display_func"]()


# Draw
# TODO: move to cv2_custom
def frame_label(
    frame, points, marker_label, font_scale=0.4, font_color=(255, 255, 255)
):
    for idx in range(len(points)):
        point = tuple(points[idx])
        tag = marker_label[idx]
        cv2_draw_label(
            frame,
            int(point[0]),
            int(point[1]),
            tag,
            fontScale=font_scale,
            fontColor=font_color,
        )


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

    marker_positions = MarkerPositions.from_yaml(config["PATHS"]["marker_positions"])
    keys = marker_positions.tags

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
        frame = scale_image(frame, scale)
        cap.release()

        # Select point using mouse event
        point = []
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
        display()
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        assert len(point) > 0, "No point selected"
        points.append(point)
    points = np.array(points)

    if glob_run_id:
        candidates = pathlib.Path(".").glob(
            config["PATHS"]["footage_video_path"].format(tag, 0, "*")
        )
        pattern = config["PATHS"]["footage_video_path"].format(tag, 0, "(\d+)")
        all_run_id = []
        for candidate in candidates:
            rr = re.search(pattern, candidate.as_posix())
            if rr is None:
                continue
            rid = int(rr.group(1))
            all_run_id.append(rid)

    # Load existing points and marker_label, and append
    for rid in all_run_id:
        initial_point_file = config["PATHS"]["tracing_data_path"].format(tag, rid)
        with OriginData(path=initial_point_file) as dataset:
            dataset.set_camera_frames(cam_id, points)

if __name__ == "__main__":
    main()
