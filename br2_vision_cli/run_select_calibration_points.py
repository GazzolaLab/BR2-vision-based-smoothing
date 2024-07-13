import glob
import os
import pathlib
import re
import sys
import time
from random import shuffle

import click
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QInputDialog, QLineEdit, QPushButton, QWidget
from sklearn import linear_model as lm
from sklearn.cluster import KMeans

import br2_vision
from br2_vision.cv2_custom.marking import cv2_draw_label
from br2_vision.cv2_custom.transformation import scale_image_from_path
from br2_vision.dlt import DLT, DLT2D, label_to_3Dcoord
from br2_vision.utility.logging import config_logging, get_script_logger


# PyQt5 Script
def prompt_dialog_integer(title: str, prompt: str):
    num, ok = QInputDialog.getInt(QWidget(), title, prompt)
    if ok:
        return num
    else:
        return None


def labeling(
    frame,
    tag,
    save_path_points,
    save_path_dlt,
    save_path_image,
    cam_id,
    x_id,
    config,
):
    """

    Use 2D-DLT method to label coordinates.

    Parameters
    ----------
    frame :
        Image to process
    tag :
        Tag for the image
    save_path_points :
        Coordinate save path. First check if the previous work exist.
    save_path_dlt :
        DLT save path. First check if the previous work exist.
    save_path_image :
        Final image path for visualizing calibration.
    cam_id :
        Camera ID
    x_id :
        frame X ID
    config :
        Configuration
    """

    # Defining mouse event handler
    _currently_selected_point_index = -1

    # Number of markers in y-z plane
    ys, ye = 1, int(config["DIMENSION"]["num_calibration_y"])
    zs, ze = 1, int(config["DIMENSION"]["num_calibration_z"])

    def onMouse(event, x, y, flags, param):
        """
        If left mouse button is clicked, either create new unlabeled coordinate or
        relocate the existing coordinate. If the clicked position is close to existing
        label, reallocate the point.
        """

        # Behavior Configuration
        MINDIST = 20  # Minimum threshold to reallocation

        # Event
        coords = param["coords"]
        window = param["window"]
        if event == cv2.EVENT_LBUTTONDOWN:  # Move coordinate
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if len(coords) > 0:
                    dist = [
                        ((_x - x) ** 2 + (_y - y) ** 2) ** 0.5
                        for _x, _y, _, _, _ in coords
                    ]
                    i = np.argmin(dist)  # Closest index
                    if dist[i] < MINDIST:
                        print("Delete point")
                        coords.pop(i)
            else:
                if len(coords) == 0:
                    coords.append([x, y, -1, -1, False])
                else:
                    dist = [
                        ((_x - x) ** 2 + (_y - y) ** 2) ** 0.5
                        for _x, _y, _, _, _ in coords
                    ]
                    i = np.argmin(dist)  # Closest index
                    if dist[i] < MINDIST:
                        coords[i] = [x, y, coords[i][2], coords[i][3], False]
                    else:  # Add new coordinate
                        coords.append([x, y, -1, -1, False])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                for coord in coords:
                    coord[4] = True
            else:
                if len(coords) > 0:
                    dist = [
                        ((_x - x) ** 2 + (_y - y) ** 2) ** 0.5
                        for _x, _y, _, _, _ in coords
                    ]
                    i = np.argmin(dist)  # Closest index
                    if dist[i] < MINDIST:
                        print("Lock current reference point: {}".format(coords[i]))
                        coords[i][4] = not coords[i][4]

    # Import existing work if there are any.
    dlt2D = DLT2D(save_path=save_path_dlt)
    dlt2D.load(pass_if_does_not_exist=True)

    coords = []  # Each item has (u, v, y-id, z-id, lock). (-1) value means unallocated
    if os.path.exists(save_path_points):
        data = np.load(save_path_points)
        for u, v, _y, _z, _lock in data["coords"]:
            coords.append([int(u), int(v), int(_y), int(_z), bool(_lock)])

    # Create GUI
    window_name = "figure - " + tag
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name, onMouse, param={"coords": coords, "window": window_name}
    )
    while True:
        # Plot
        image = frame.copy()

        # Locate labels
        for u, v, y, z, l in coords:
            color = (270, 20, 20) if not l else (20, 20, 270)
            cv2_draw_label(image, u, v, (y, z), color=color)

        # Draw
        cv2.imshow(window_name, image)

        # Keyboard Options
        key = cv2.waitKey(1) & 0xFF
        if key == 32 or key == 27 or key == 13:  # Exit (13: enter, 32: space, 27: ESC)
            break
        elif key in [ord("h"), ord("H")]:  # help
            print("key:{} - help message".format(key))
            print(
                """
- Left Click: Select point
    - Control-Left Click: Delete point
- Right Click: Lock point
    - Control-Right Click: Lock all points
- Key 'h' or 'H': Show help
- Key 'D' or 'd': Delete last coordinate
- Key 'b': Label points
- Key 'p': Interpolate using planar(2D) DLT (At least 4 locked points are required)
- Key 'P': Use Harry's corner detection.
- Key 'o': Use 3D DLT (from other reference frame images)
- Key 's': Save
- 'Enter,' 'Space,' 'ESC': Complete, move-on to next frame
            """
            )
        elif key == ord("d"):  # Delete last coordinate
            print("key:{} - Delete last item".format(key))
            if len(coords) > 0:
                coords.pop(-1)
        elif key == ord("D"):  # Delete last coordinate
            print("key:{} - Delete all item".format(key))
            while len(coords) > 0:
                coords.pop(-1)
        elif key == ord("b"):  # Labeling process
            if len(coords) == 0:
                print("No unlabelled point")
            else:
                for i in range(len(coords)):
                    u, v, y, z, locked = coords[i]
                    if locked:
                        continue
                    if (y, z) == (-1, -1):  # need new label
                        image = frame.copy()
                        cv2_draw_label(image, u, v, (y, z))
                        cv2.imshow(window_name, image)
                        cv2.waitKey(1)
                        _y = int(input("Which y-position: "))
                        _z = int(input("Which z-position: "))
                        coords[i][2] = _y
                        coords[i][3] = _z
                print("Labeling done.")
        elif key == ord("p") or key == ord("P"):  # Populate the interpolated points.
            # If P is pressed, use corner detection
            if len(coords) < 4:
                print(
                    "Need at least 4 points to draw estimation. (we have {})".format(
                        len(coords)
                    )
                )
                continue
            elif (-1, -1) in [(c[2], c[3]) for c in coords]:
                print("Please label all points (press l)")
                continue

            # 2d dlt
            dlt2D.clear()
            if ys is None or ye is None or zs is None or ze is None:
                continue
            _ys, _ye, _zs, _ze = ys, ye, zs, ze  # Save the parameter

            for u, v, y_id, z_id, locked in coords:
                if not locked:
                    continue
                _, y, z = label_to_3Dcoord(0, y_id, z_id, config)
                dlt2D.add_reference(u, v, y, z)
            try:
                dlt2D.finalize()
            except AssertionError:
                print("Need more points")
                continue

            locked_coords = [
                [u, v, y_id, z_id, locked]
                for u, v, y_id, z_id, locked in coords
                if locked
            ]
            locked_id = [(y_id, z_id) for _, _, y_id, z_id, locked in coords if locked]
            coords.clear()
            coords.extend(locked_coords)
            width = int(frame.shape[1])
            height = int(frame.shape[0])
            for y_id in range(ys, ye + 1):
                for z_id in range(zs, ze + 1):
                    if (y_id, z_id) in locked_id:
                        continue
                    _, y, z = label_to_3Dcoord(0, y_id, z_id, config)
                    u, v = dlt2D.inverse_map(y, z)
                    if u >= 0 and u < width and v >= 0 and v < height:
                        if key == ord("P"):  # corner detection
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(
                                np.float32
                            )
                            term = (
                                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                                100,
                                0.1,
                            )
                            _pts = np.array([u, v], dtype=np.float32).reshape([1, 1, 2])
                            _pts2 = cv2.cornerSubPix(
                                gray, _pts, (19, 19), (-1, -1), term
                            )
                            u, v = _pts2.reshape([2]).astype(int)
                        coords.append([u, v, y_id, z_id, False])

            dlt2D.save()
            print("finished drawing")
        elif key == ord("o"):  # 3d:
            calibration_ref_point_save_wild = config["PATHS"][
                "calibration_ref_point_save_wild"
            ]
            reference_point_filenames = glob.glob(
                calibration_ref_point_save_wild.format(cam_id)
            )
            if len(reference_point_filenames) < 2:
                print(
                    "At least 2 frames must be calibrated before using 3d inverse-dlt"
                )
                continue
            _3d_dlt = DLT(calibration_path=config["PATHS"]["calibration_path"])
            _3d_dlt.add_camera(cam_id, calibration_type=11)
            for filename in reference_point_filenames:
                _x_id = int(filename.split("-")[-2])
                data = np.load(filename)["coords"]

                for u, v, y_id, z_id, _ in data:
                    x, y, z = label_to_3Dcoord(_x_id, y_id, z_id, config)
                    _3d_dlt.add_reference(u, v, x, y, z, camera_id=cam_id)
            _3d_dlt.finalize()

            locked_coords = [
                [u, v, y_id, z_id, locked]
                for u, v, y_id, z_id, locked in coords
                if locked
            ]
            locked_id = [(y_id, z_id) for _, _, y_id, z_id, locked in coords if locked]
            coords.clear()
            coords.extend(locked_coords)
            width = int(frame.shape[1])
            height = int(frame.shape[0])
            for y_id in range(ys, ye + 1):
                for z_id in range(zs, ze + 1):
                    if (y_id, z_id) in locked_id:
                        continue
                    x, y, z = label_to_3Dcoord(x_id, y_id, z_id, config)
                    u, v = _3d_dlt.inverse_map(x, y, z)[cam_id]
                    u = int(u)
                    v = int(v)
                    if u >= 0 and u < width and v >= 0 and v < height:
                        coords.append([u, v, y_id, z_id, False])
            print("finished 3d dlt")
        elif key == ord("s"):  # Save
            print("key:{} - Save points".format(key))
            np.savez(save_path_points, coords=coords)

    # Presave the result
    np.savez(save_path_points, coords=coords)
    print(f"{save_path_image=}")
    print(f"{image.shape=}")
    cv2.imwrite(save_path_image, image)
    cv2.destroyAllWindows()

    return coords


@click.command()
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose")
@click.option(
    "-d",
    "--dry",
    is_flag=True,
    default=False,
    help="Dry run: print table of processing frames.",
)
@click.option("-S", "--show", is_flag=True, default=False, help="Show frames")
def select_calibration_points(verbose, dry, show):
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))
    scale = float(config["DIMENSION"]["scale_video"])

    app = QApplication([])

    raw_videos = []  # [(cam_id, video_path)]
    calibration_video_wild = config["PATHS"]["calibration_video"].format("*")
    collections = glob.glob(calibration_video_wild, recursive=True)
    for path in collections:
        s = re.findall(r"cam\d+", path)[0][3:]
        assert (
            s.isdigit()
        ), f"Camera id must be a number, and filepath name must be cam{{id}}.MOV"
        raw_videos.append((int(s), path))

    reference_image_paths = []  # (Camera ID, x-location ID, image path)
    for cam_id, video_path in raw_videos:
        logger.debug("Selecting from video file: {}".format(video_path))
        video_path = pathlib.Path(video_path)
        directory = video_path.parent / video_path.stem

        k = directory / f"*.{config['DEFAULT']['processing_image_extension']}"
        images = glob.glob(k.as_posix())
        images = sorted(images)

        if len(images) == 0:
            raise ValueError(f"No frame extracted for the video: {video_path}")

        for xid, image_path in enumerate(images):
            reference_image_paths.append((cam_id, xid, image_path))
    reference_image_paths = sorted(reference_image_paths)

    if dry:
        # Print table
        for camera_id, x_id, path in reference_image_paths:
            print(f"{camera_id=}\t{x_id=}\t{path}")
        return

    # Create directory
    calibration_ref_path = config["PATHS"]["calibration_ref_points_path"]
    os.makedirs(calibration_ref_path, exist_ok=True)

    # Label Reference Point
    calibration_ref_point_save = config["PATHS"]["calibration_ref_point_save"]
    calibration_dlt_path = config["PATHS"]["calibration_dlt_path"]
    calibration_view_path = config["PATHS"]["calibration_view_path"]
    for camera_id, x_id, path in reference_image_paths:
        logger.info(f"Processing camera {camera_id} at xid={x_id}: {path}")
        frame = scale_image_from_path(path, scale=scale)
        points = labeling(
            frame=frame,
            tag=path,
            save_path_points=calibration_ref_point_save.format(camera_id, x_id),
            save_path_dlt=calibration_dlt_path.format(camera_id, x_id),
            save_path_image=calibration_view_path.format(camera_id, x_id),
            cam_id=camera_id,
            x_id=x_id,
            config=config,
        )
        print("CAM {} Points:".format(camera_id))
        print(points)


if __name__ == "__main__":
    select_calibration_points()
