import os, sys
import numpy as np
import pathlib
import cv2
import matplotlib.pyplot as plt

from cv2_custom.marking import cv2_draw_label
from cv2_custom.transformation import scale_image

#from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QInputDialog

import click

import br2_vision
from br2_vision.utility.logging import config_logging, get_script_logger
from br2_vision.data import MarkerPositions, TrackingData, FlowQueue

def zoomed_inquiry(current_frame, uv, scale=5.0, disp_h=80, disp_w=80):
    x, y = uv
    x = int(x)
    y = int(y)

    # Region of interest display
    window_name_roi = "roi"
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

    def onMouse(event, x, y, flags, param):
        uv = param["uv"]
        original_uv = param["original_uv"]
        if event == cv2.EVENT_LBUTTONDOWN:
            uv[0] = x
            uv[1] = y
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Return original uv
            uv[:] = original_uv

    # Inquiry Loop
    inquiry_on = True
    window_name = "select reappeared point"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name, onMouse, param={"uv": _uv, "original_uv": original_uv}
    )
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
    tags = param["tags"]
    bypass_inquiry = flags & cv2.EVENT_FLAG_CTRLKEY
    if event == cv2.EVENT_LBUTTONDOWN:
        point = np.array([x, y], dtype=np.int32).reshape([1, 2])
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Second zoom-layer selection
        uv = zoomed_inquiry(param["frame"], np.array([x, y]))
        point = uv.astype(np.int32).reshape([1, 2])
    else:
        return
    points.append(point)

    # Ask for a tag in a separate window
    if bypass_inquiry:
        tag = prev_tag
    else:
        _ok = False
        while not _ok:
            tag, _ok = QInputDialog.getText(
                QWidget(), "Tag", "Input Tag", text=prev_tag
            )
    if tag[0] == "R":
        prev_tag = (
            tag[0]
            + str(int(tag.split("-")[0][1:]) + 1)
            + "-"
            + str(int(tag.split("-")[1]))
        )
    else:
        prev_tag = tag
    tags.append(tag)
    print("added: ")
    print(point, tag)

# Draw
def frame_label(frame, points, tags):
    for inx in range(len(points)):
        point = tuple(points[inx][0])
        tag = tags[inx]
        cv2_draw_label(frame, int(point[0]), int(point[1]), tag, fontScale=0.8)

@click.command()
@click.option(
    "-t",
    "--tag",
    type=str,
    help="Experiment tag. Path ./tag should exist.",
)
@click.option(
    "-c", "--cam-id", type=int, help="Camera index given in file.", multiple=True
)
@click.option(
    "-r", "--run-id", type=int, help="Specify run index. Initial points are saved for all specified run-ids.", multiple=True
)
@click.option(
    "-ss", "--start-frame", type=int, help="Start frame.", default=0
)
@click.option(
    "-es", "--end-frame", type=int, help="End frame.", default=-1
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def main(tag, cam_id, run_id, start_frame, end_frame, verbose, dry):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    if len(run_id) > 1 and start_frame != 0:
        logger.error("Start frame is only supported for single run_id.")
        sys.exit(1)

    marker_positions = MarkerPositions.from_yaml(config["PATHS"]["marker_positions"])
    keys = marker_positions.keys()

    # Set Colors
    _N = 100
    np.random.seed(100)
    color = np.random.randint(0, 235, (100, 3)).astype(int)

    # Path
    for cid in cam_id:
        video_path = config["PATHS"]["footage_video"].format(tag, cam_id, run_id[0])
        initial_point_file = config["PATHS"]["tracing_data_path"]

        video_name = os.path.basename(video_path)

        # Capture Video
        cap = cv2.VideoCapture(os.path.join(path, video_name))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame == -1:
            start_frame = video_length - 1
        if end_frame == -1:
            end_frame = video_length - 1
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, curr_frame = cap.read()

        assert start_frame < video_length

        # app = QApplication(sys.argv)
        tags = []
        points = []

        # First-layer Selection
        cv2.namedWindow(video_name)
        cv2.setMouseCallback(
            video_name,
            mouse_event_click_point,
            param={"frame": curr_frame, "points": points, "tags": tags},
        )
        while True:
            disp_img = curr_frame.copy()

            if len(points) > 0:
                frame_label(disp_img, points, tags)

            cv2.imshow(video_name, disp_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                print("done")
                break
            elif key == ord("d"):
                if len(points) > 0:
                    points.pop(-1)
                    tags.pop(-1)
                    print("deleted")
            elif key == ord("p"):
                print("check")
                print(points)
                print(tags)
                print('')
                print("c: complete")
                print("d: delete last point")
        cv2.destroyAllWindows()

        # Load existing points and tags
        for rid in run_id:
            with TrackingData.initialize(path=initial_point_file.format(tag, rid), marker_positions=marker_positions) as dataset:
                for tag, point in zip(tags, points):
                    flow_queue = FlowQueue(tag, point, start_frame, end_frame, cid)
                    dataset.append(flow_queue)

    visualize(initial_piont_file, tag, cam_id, run_id, config)

def visualize(data_path, tag, cam_id, run_id, config, frame=0):
    data_path = config["PATHS"]["tracing_data_path"]
    working_dir = pathlib.Path(config["PATHS"]["postprocessing_path"].format(tag))

    initial_points_dir = working_dir / "initial_points"
    initial_points_dir.mkdir(parents=True, exist_ok=True)
    
    for rid in run_id:
        with TrackingData.load(path=data_path.format(tag, rid)) as dataset:
            for cid in cam_id:
                plt.figure()
                flow_queues = dataset.get_flow_queues(camera=cid, start_frame=frame)
                points = np.array([queue.point for queue in flow_queues])  # (N, 2)
                plt.scatter(points[:, 0], points[:, 1])
                plt.title(f'frame {frame}')
                plt.savefig(initial_points_dir / f"initial_points_cam{cid}_run{rid}_frame{frame}.png")
                plt.close('all')

if __name__ == "__main__":
    main()
