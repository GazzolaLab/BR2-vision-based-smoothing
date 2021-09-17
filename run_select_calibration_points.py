import os
import sys
import glob
from collections import defaultdict
from random import shuffle 

import numpy as np
import cv2

from sklearn import linear_model as lm
from sklearn.cluster import KMeans

from dlt import DLT2D, DLT

from cv2_custom.marking import cv2_draw_label

from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLineEdit, QInputDialog)

from config import *

# Config - Search for all calibration images
IMAGE_PATH = CALIBRATION_PATH
calibration_images = glob.glob(os.path.join(IMAGE_PATH, 'cam-*-calibration-*[0-9].png'))
calibration_images.sort(key=lambda x: (int(x.split('-')[1]), int(x.split('-')[-1].split('.')[0])) )
reference_image_paths = {}  # key: (Camera ID, x-location ID)
for path in calibration_images:
    base = os.path.basename(path)
    base = base.split('.')[0].split('-')
    reference_image_paths[(base[1], base[3])] = path
OUTPUT_NAME = CALIBRATION_REF_POINTS_PATH

# Label to 3d coordinate
def label_to_3Dcoord(x_label:int, y_label:int, z_label:int):
    x_label = int(x_label)
    y_label = int(y_label)
    z_label = int(z_label)
    delta = np.array([0.020, 0.040, 0.040])  # Distance between interval in (xyz)
    id_vec = np.array([x_label, y_label, z_label], dtype=float)
    return id_vec * delta

# PyQt5 Script
def prompt_dialog_integer(title:str, prompt:str):
    num, ok = QInputDialog.getInt(QWidget(), title, prompt)
    if ok:
        return num
    else:
        return None

# CV2 Script
def scale_image(filepath, scale=1.0):
    # Extract Control Frame
    frame = cv2.imread(filepath)
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame

def labeling(frame, tag, save_path_points, save_path_dlt, save_path_image, cam_id, x_id):
    """

    Use 2D-DLT method to label coordinates.

    Parameters
    ----------
    frame :
        Image to process
    save_path_dlt :
        DLT save path. First check if the previous work exist.
    save_path_image :
        Final image path for visualizing calibration.
    """

    # Defining mouse event handler
    _currently_selected_point_index = -1
    def onMouse(event, x, y, flags, param):
        """
        If left mouse button is clicked, either create new unlabeled coordinate or
        relocate the existing coordinate. If the clicked position is close to existing
        label, reallocate the point.
        """

        # Behavior Configuration
        MINDIST = 20  # Minimum threshold to reallocation

        # Event
        coords = param['coords']
        window = param['window']
        if event == cv2.EVENT_LBUTTONDOWN: # Move coordinate
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if len(coords) > 0:
                    dist = [((_x-x)**2+(_y-y)**2)**0.5 for _x, _y, _, _, _ in coords]
                    i = np.argmin(dist)  # Closest index
                    if dist[i] < MINDIST:
                        print('Delete point')
                        coords.pop(i)
            else:
                if len(coords) == 0:
                    coords.append([x,y,-1,-1,False])
                else:
                    dist = [((_x-x)**2+(_y-y)**2)**0.5 for _x, _y, _, _, _ in coords]
                    i = np.argmin(dist)  # Closest index
                    if dist[i] < MINDIST:
                        coords[i] = [x, y, coords[i][2], coords[i][3], False]
                    else: # Add new coordinate
                        coords.append([x,y,-1,-1,False])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                for coord in coords:
                    coord[4] = True
            else:
                if len(coords) > 0:
                    dist = [((_x-x)**2+(_y-y)**2)**0.5 for _x, _y, _, _, _ in coords]
                    i = np.argmin(dist)  # Closest index
                    if dist[i] < MINDIST:
                        print('Lock current reference point: {}'.format(coords[i]))
                        coords[i][4] = not coords[i][4]

    # Import existing work if there are any.
    dlt2D = DLT2D(save_path=save_path_dlt)
    dlt2D.load(pass_if_does_not_exist=True)

    coords = []  # Each item has (u, v, y-id, z-id, lock). (-1) value means unallocated
    if os.path.exists(save_path_points):
        data = np.load(save_path_points)
        for u, v, _y, _z, _lock in data['coords']:
            coords.append([int(u), int(v), int(_y), int(_z), bool(_lock)])

    # Create GUI
    window_name = 'figure - '+tag
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, onMouse,
        param={'coords': coords, 'window': window_name})
    while True:
        # Plot
        image = frame.copy()

        # Locate labels
        for u, v, y, z, l in coords:
            color = (270,20,20) if not l else (20,20,270)
            cv2_draw_label(image, u, v, (y,z), color=color)

        # Draw
        cv2.imshow(window_name, image)

        # Keyboard Options
        key = cv2.waitKey(1) & 0xFF
        if key == 32 or key == 27 or key == 13:  # Exit (13: enter, 32: space, 27: ESC)
            break
        elif key == ord("d"):  # Delete last coordinate
            print('key:{} - Delete last item'.format(key))
            if len(coords) > 0:
                coords.pop(-1)
        elif key == ord("D"):  # Delete last coordinate
            print('key:{} - Delete all item'.format(key))
            while len(coords) > 0:
                coords.pop(-1)
        elif key == ord("b"):  # Labeling process
            if len(coords) == 0:
                print('No unlabelled point')
            else:
                for i in range(len(coords)):
                    u, v, y, z, locked = coords[i]
                    if locked:
                        continue
                    if (y, z) == (-1,-1): # need new label
                        image = frame.copy()
                        cv2_draw_label(image, u, v, (y,z))
                        cv2.imshow(window_name, image)
                        cv2.waitKey(1)
                        _y = int(input('Which y-position (vertical): '))
                        _z = int(input('Which z-position (horizontal): '))
                        coords[i][2] = _y
                        coords[i][3] = _z
                print('Labeling done.')
        elif key == ord("p") or key == ord("P"):  # Populate the interpolated points.
            # If P is pressed, use corner detection
            if len(coords) < 4:
                print('Need at least 4 points to draw estimation. (we have {})'
                        .format(len(coords)))
                continue
            elif (-1,-1) in [(c[2], c[3]) for c in coords]:
                print('Please label all points (press l)')
                continue

            # 2d dlt
            dlt2D.clear()
            ys, ye, zs, ze = 1,12,1,10 # Number of markers in y-z plane
            if ys is None or ye is None or zs is None or ze is None:
                continue
            _ys, _ye, _zs, _ze = ys, ye, zs, ze # Save the parameter

            for u, v, y_id, z_id, locked in coords:
                if not locked:
                    continue
                _, y, z = label_to_3Dcoord(0, y_id, z_id)
                dlt2D.add_reference(u, v, y, z)
            try:
                dlt2D.finalize()
            except AssertionError:
                print('Need more points')
                continue

            locked_coords = [[u,v,y_id,z_id,locked] for u,v,y_id,z_id,locked in coords if locked]
            locked_id = [(y_id, z_id) for _,_,y_id,z_id,locked in coords if locked]
            coords.clear()
            coords.extend(locked_coords)
            width = int(frame.shape[1])
            height = int(frame.shape[0])
            for y_id in range(ys,ye+1):
                for z_id in range(zs,ze+1):
                    if (y_id, z_id) in locked_id:
                        continue
                    _, y, z = label_to_3Dcoord(0, y_id, z_id)
                    u, v = dlt2D.inverse_map(y, z)
                    if u >= 0 and u < width and v >= 0 and v < height:
                        if key == ord("P"):  # corner detection
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 0.1)
                            _pts = np.array([u,v], dtype=np.float32).reshape([1,1,2])
                            _pts2 = cv2.cornerSubPix(gray, _pts, (19,19), (-1,-1), term)
                            u, v = _pts2.reshape([2]).astype(int)
                        coords.append([u, v, y_id, z_id, False])

            dlt2D.save()
            print('finished drawing')
        elif key == ord("o"): # 3d:
            reference_point_filenames = glob.glob(CALIBRATION_REF_POINT_SAVE_WILD.format(cam_id))
            if len(reference_point_filenames) < 2:
                print('At least 2 frames must be calibrated before using 3d inverse-dlt')
                continue
            _3d_dlt = DLT()
            _3d_dlt.add_camera(cam_id, calibration_type=11)
            for filename in reference_point_filenames:
                _x_id = int(filename.split('-')[-2])
                data = np.load(filename)['coords']

                for u, v, y_id, z_id, _ in data:
                    x, y, z = label_to_3Dcoord(_x_id, y_id, z_id)
                    _3d_dlt.add_reference(u,v,x,y,z,camera_id=cam_id)
            _3d_dlt.finalize()

            locked_coords = [[u,v,y_id,z_id,locked] for u,v,y_id,z_id,locked in coords if locked]
            locked_id = [(y_id, z_id) for _,_,y_id,z_id,locked in coords if locked]
            ys, ye, zs, ze = 1,12,1,10
            coords.clear()
            coords.extend(locked_coords)
            width = int(frame.shape[1])
            height = int(frame.shape[0])
            for y_id in range(ys,ye+1):
                for z_id in range(zs,ze+1):
                    if (y_id, z_id) in locked_id:
                        continue
                    x, y, z = label_to_3Dcoord(x_id, y_id, z_id)
                    u, v = _3d_dlt.inverse_map(x, y, z)[cam_id]
                    u = int(u)
                    v = int(v)
                    if u >= 0 and u < width and v >= 0 and v < height:
                        coords.append([u, v, y_id, z_id, False])
            print('finished 3d dlt')
        elif key == ord("s"):  # Save
            print('key:{} - Save points'.format(key))
            np.savez(save_path_points, coords=coords)

    # Presave the result
    np.savez(save_path_points, coords=coords)
    cv2.imwrite(save_path_image, image)
    cv2.destroyAllWindows()

    return coords


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Label Reference Point
    results = defaultdict(list)
    for (camera_id, x_id), path in reference_image_paths.items():
        frame = scale_image(path, scale=1.0)
        points = labeling(
            frame=frame,
            tag=path,
            save_path_points=CALIBRATION_REF_POINT_SAVE.format(camera_id, x_id),
            save_path_dlt=CALIBRATION_DLT_PATH.format(camera_id, x_id),
            save_path_image=CALIBRATION_VIEW_PATH.format(camera_id, x_id),
            cam_id=camera_id,
            x_id=x_id,
        )
        print('CAM {} Points:'.format(camera_id))

        # Save the points
        for u, v, y_id, z_id, _ in points:
            x, y, z = label_to_3Dcoord(x_id, y_id, z_id)
            u = int(u)
            v = int(v)
            results[camera_id].append((u,v,x,y,z))

    np.savez(OUTPUT_NAME, **results)
