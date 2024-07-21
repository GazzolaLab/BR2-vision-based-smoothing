import os
import math

import cv2
import numpy as np
import pickle

import click

# mouse callback function
def handle_mouse_event(event, x, y, flags, param):
    points: list[tuple(float, float)] = param["points"]
    window_name = param["window_name"]
    frame = param["frame"]

    state_changed = True
    if event == cv2.EVENT_LBUTTONDOWN:
        # add the point to the list
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # snap the nearest point to the mouse location
        if len(points) == 0:
            return
        distances = [math.hypot(p[0] - x, p[1] - y) for p in points]
        min_index = np.argmin(distances)
        points[min_index] = (x, y)
    else:
        state_changed = False
       
    if state_changed:
        _frame = add_points_to_image(frame, points)
        cv2.imshow(window_name, _frame)

def add_points_to_image(image, points):
    if not points:
        return
    image = image.copy()
    for point in points:
        cv2.circle(image, point, 5, (0, 240, 0), -1)
    return image

# draw the tangent line
def draw_tangent(image, x_points, y_points, color=(0, 0, 255)):
    A = np.vstack([x_points, np.ones(len(x_points))]).T
    slope, intercept = np.linalg.lstsq(A, y_points, rcond=None)[0]
    height, width = image.shape[:2]
    p1 = (0, int(intercept))
    p2 = (width, int(slope * width + intercept))
    cv2.line(image, p1, p2, color, 2)

def resize_image(image, width):
    height = int(image.shape[0] * (width / image.shape[1]))
    return cv2.resize(image, (width, height))

def crop_image(image, roi):
    x, y, w, h = roi
    return image[y:y+h, x:x+w]

@click.command()
@click.option('-f', '--file_path', type=click.Path(exists=True), help='Path to the image file')
@click.option('-l', '--length', type=float, help='Length of the rod')
def main(file_path, length):
    # Load video
    cap = cv2.VideoCapture(file_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()

    # Select ROI : on resized image
    width = 1000
    cache_path = f"{file_path}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            points = data["points"]
            roi = data["roi"]
            current_frame = data.get("current_frame", 0)
    else:
        points = []
        roi = cv2.selectROI("Select ROI", resize_image(frame, width))
        cv2.destroyAllWindows()
        current_frame = 0

    frame = crop_image(resize_image(frame, width), roi)

    window_name = "Select Points"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handle_mouse_event, {"points": points, "window_name": window_name, "frame": frame})
    while True:
        # Navigate to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, fetch_frame = cap.read()
        if not ret:
            break
        frame[:] = crop_image(resize_image(fetch_frame, width), roi)

        cv2.imshow(window_name, frame)

        # api
        # x: exit
        # q: -5 frames, w: -1 frame, e: +1 frame, r: +5 frames
        # a: beginning, f: last
        # s: save cache
        # d: reset
        key = cv2.waitKey(0)
        if key == ord("x"):
            break
        elif key == ord("a"):
            current_frame = 0
            print(f"{current_frame}/{total_frame}")
        elif key == ord("f"):
            current_frame = total_frame - 1
            print(f"{current_frame}/{total_frame}")
        elif key == ord("q"):
            current_frame = max(0, current_frame - 5)
            print(f"{current_frame}/{total_frame}")
        elif key == ord("w"):
            current_frame = max(0, current_frame - 1)
            print(f"{current_frame}/{total_frame}")
        elif key == ord("e"):
            current_frame = min(total_frame - 1, current_frame + 1)
            print(f"{current_frame}/{total_frame}")
        elif key == ord("r"):
            current_frame = min(total_frame - 1, current_frame + 5)
            print(f"{current_frame}/{total_frame}")
        elif key == ord("d"):
            points = []
            cv2.imshow(window_name, frame)
        elif key == ord("s"):
            with open(cache_path, "wb") as f:
                pickle.dump(points, f)
    cap.release()
    cv2.destroyAllWindows()

    points = np.asarray(points) # (N, 2)
    vectors = np.diff(points, axis=0)
    spacing = np.linalg.norm(vectors, axis=1)

    # Find cosine angles between the vectors
    cosine_angles = (vectors[:-1] * vectors[1:]).sum(axis=1) / (spacing[:-1] * spacing[1:])
    angles = np.arccos(cosine_angles)
    total_angle = angles.sum()
    print(f"Total angle: {total_angle} rad")

    # Find Average Curvature
    x, y = points.T * length / spacing.sum()
    dx = np.gradient(x, x)  # first derivatives
    dy = np.gradient(y, x)

    d2x = np.gradient(dx, x)  # second derivatives
    d2y = np.gradient(dy, x)

    cur = np.abs(d2y) / (np.sqrt(1 + dy ** 2)) ** 1.5  # curvature
    cur[np.isnan(cur)] = 0.
    max_curvature = cur.max()
    avg_curvature = cur.mean()
    print(f"Max Curvature: {max_curvature}")
    print(f"Average Curvature: {avg_curvature}")

    with open(cache_path, "wb") as f:
        pickle.dump({"points": points, "roi": roi, "current_frame": current_frame}, f)

    print(f"{total_angle} {max_curvature} {avg_curvature}")
