import os, sys
import numpy as np
import pathlib
import glob
import cv2
import re
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QInputDialog

import click

import br2_vision
from br2_vision.utility.logging import config_logging, get_script_logger
from br2_vision.data_structure import MarkerPositions, TrackingData, FlowQueue
from br2_vision.cv2_custom.marking import cv2_draw_label
from br2_vision.cv2_custom.transformation import scale_image
from br2_vision.qt_custom.label_prompt import LabelPrompt


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

    return np.array([x, y], dtype=int)


# Mouse Handle
prev_tag = ""


def mouse_event_click_point(event, x, y, flags, param):
    global prev_tag
    points = param["points"]
    marker_label = param["marker_label"]
    old_marker_label = param["old_marker_label"]
    bypass_inquiry = flags & cv2.EVENT_FLAG_CTRLKEY
    if event == cv2.EVENT_LBUTTONDOWN:
        point = np.array([x, y], dtype=np.int32).reshape([1, 2])
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Second zoom-layer selection
        uv = zoomed_inquiry(param["frame"], np.array([x, y]))
        point = uv.astype(np.int32)
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

    if tag in old_marker_label:
        print(f"tag {tag} already exist.")
    else:
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
    "-c", "--cam-id", type=int, help="Camera index given in file."
)
@click.option(
    "-r",
    "--run-id",
    type=int,
    help="Specify run index. Initial points are saved for all specified run-ids.",
    multiple=True,
)
@click.option(
    "--glob-run-id",
    type=bool,
    default=False,
    is_flag=True,
    help="If specified, use all run-ids in the directory. If specified, --run-id is ignored.",
)
@click.option(
    "-ss", "--start-frame", type=int, help="Start frame.", default=0, show_default=True
)
@click.option(
    "-es", "--end-frame", type=int, help="End frame.", default=-1, show_default=True
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def main(tag, cam_id, run_id, glob_run_id, start_frame, end_frame, verbose, dry):
    app = QApplication(sys.argv)  # Initialize Q application

    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))
    scale = float(config["DIMENSION"]["scale_video"])

    if glob_run_id:
        candidates = pathlib.Path(".").glob(config["PATHS"]["footage_video_path"].format(tag, cam_id, "*"))
        #candidates = glob.glob(config["PATHS"]["footage_video_path"].format(tag, cam_id, "*"))
        pattern = config["PATHS"]["footage_video_path"].format(tag, cam_id, "(\d+)")
        run_id = []
        for candidate in candidates:
            rr = re.search(pattern, candidate.as_posix())
            if rr is None:
                continue
            rid = int(rr.group(1))
            run_id.append(rid)

    if len(run_id) > 1 and start_frame != 0:
        logger.error("Start frame is only supported for single run_id.")
        sys.exit(1)

    marker_positions = MarkerPositions.from_yaml(config["PATHS"]["marker_positions"])
    keys = marker_positions.tags

    # Set prompt
    prompt = LabelPrompt(
        app,
        list(map(str, range(len(marker_positions)))),
        keys,
        "cross-section index",
        "label tag",
    )

    # Set Colors
    _N = 100
    np.random.seed(100)
    color = np.random.randint(0, 235, (100, 3)).astype(int)

    # Path
    video_path = config["PATHS"]["footage_video_path"].format(tag, cam_id, run_id[0])
    assert os.path.exists(video_path), f"Video not found: {video_path}."
    initial_point_file = config["PATHS"]["tracing_data_path"].format(tag, run_id[0])

    with TrackingData.initialize(
        path=initial_point_file, marker_positions=marker_positions
    ) as dataset:
        flow_queues = dataset.get_flow_queues(camera=cam_id, start_frame=start_frame, force_run_all=True)
        old_points = [queue.point for queue in flow_queues]  # (N, 2)
        old_marker_label = [
            (queue.z_index, queue.label) for queue in flow_queues
        ]  # (N, 2)

    video_name = os.path.basename(video_path)

    # Capture Video
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, curr_frame = cap.read()
    curr_frame = scale_image(curr_frame, scale)

    assert start_frame < video_length

    marker_label = []
    points = []

    # TODO: refactor
    def display():
        disp_img = curr_frame.copy()

        if len(old_points) > 0:
            frame_label(
                disp_img, old_points, old_marker_label, font_color=(0, 255, 0)
            )
        if len(points) > 0:
            frame_label(disp_img, points, marker_label)

        cv2.imshow(video_name, disp_img)

    # First-layer Selection
    cv2.namedWindow(video_name)
    cv2.setMouseCallback(
        video_name,
        mouse_event_click_point,
        param={
            "frame": curr_frame,
            "points": points,
            "marker_label": marker_label,
            "old_marker_label": old_marker_label,
            "prompt": prompt,
            "display_func": display,
        },
    )
    while True:
        display()

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            print("done")
            break
        elif key == ord("d"):
            if len(points) > 0:
                print(f"deleted: {points[-1]} -> {marker_label[-1]}")
                points.pop(-1)
                marker_label.pop(-1)
        elif key == ord("h"):
            print("check")
            print(f"preloaded markers: {points=}, {marker_label=}")
            print(points)
            print(marker_label)
            print("")
            print("c: complete")
            print("d: delete last point")
    cv2.destroyAllWindows()

    # Load existing points and marker_label, and append
    # TODO: Maybe use loaded queue from the beginning
    for rid in run_id:
        video_path = config["PATHS"]["footage_video_path"].format(tag, cam_id, rid)
        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if end_frame == -1:
            _end_frame = video_length

        initial_point_file = config["PATHS"]["tracing_data_path"].format(tag, rid)
        with TrackingData.initialize(
            path=initial_point_file,
            marker_positions=marker_positions,
        ) as dataset:
            # print(f"{len(dataset.get_flow_queues(camera=cam_id))=}")
            for label, point in zip(marker_label, points):
                point = tuple(point.ravel().tolist())
                flow_queue = FlowQueue(
                    point, start_frame, _end_frame, cam_id, label[0], label[1]
                )
                dataset.append(flow_queue)
            if run_id[0] == rid:  # TODO: fix this!!
                for label, point in zip(old_marker_label, old_points):
                    point = tuple(point.ravel().tolist())
                    flow_queue = FlowQueue(
                        point, start_frame, _end_frame, cam_id, label[0], label[1]
                    )
                    dataset.append(flow_queue)

    visualize(tag, cam_id, run_id, config)

    app.quit()  # Quit Q application


def visualize(tag, cam_id, run_id, config, frame=0):
    initial_point_file = config["PATHS"]["tracing_data_path"]
    working_dir = pathlib.Path(config["PATHS"]["postprocessing_path"].format(tag))

    initial_points_dir = working_dir / "initial_points"
    initial_points_dir.mkdir(parents=True, exist_ok=True)

    for rid in run_id:
        with TrackingData.load(path=initial_point_file.format(tag, rid)) as dataset:
            plt.figure()
            flow_queues = dataset.get_flow_queues(camera=cam_id, start_frame=frame)
            if len(flow_queues) == 0:
                continue
            points = np.array([queue.point for queue in flow_queues])  # (N, 2)
            plt.scatter(points[:, 0], points[:, 1])
            plt.title(f"frame {frame}")
            plt.savefig(
                initial_points_dir
                / f"initial_points_cam{cam_id}_run{rid}_frame{frame}.png"
            )
            plt.close("all")


if __name__ == "__main__":
    main()
