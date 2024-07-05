import math
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np

# store the selected points
points = []


# calulate the last ten points and the first ten points to fit the line
def fit_line_to_points(x_points, y_points):
    A = np.vstack([x_points, np.ones(len(x_points))]).T
    m, c = np.linalg.lstsq(A, y_points, rcond=None)[0]
    return m, c


# draw the tangent line
def draw_tangent(image, x_points, y_points, color=(0, 0, 255)):
    slope, intercept = fit_line_to_points(x_points, y_points)
    height, width = image.shape[:2]
    p1 = (0, int(intercept))
    p2 = (width, int(slope * width + intercept))
    cv2.line(image, p1, p2, color, 2)


def cal_bend_cur_auto(lower_bound=30, upper_bound=180):
    root = tk.Tk()
    root.withdraw()

    # Open file dialog to select image
    file_path = filedialog.askopenfilename()

    # Load image
    image = cv2.imread(file_path)

    # Perform image cropping based on selected region

    # Resize the image
    width = 500
    height = int(image.shape[0] * (width / image.shape[1]))
    resized_image = cv2.resize(image, (width, height))

    # Select ROI
    roi = cv2.selectROI("Select ROI", resized_image)

    # get the ROI in the original image (if you need the ROI in the original resolution)
    x, y, w, h = roi
    cropped_image = image[
        int(y * (image.shape[0] / height)) : int((y + h) * (image.shape[0] / height)),
        int(x * (image.shape[1] / width)) : int((x + w) * (image.shape[1] / width)),
    ]
    # Resize the cropped image to the same size as the original image
    cropped_image = cv2.resize(cropped_image, (width, height))

    # Display the cropped image
    cv2.imshow("Cropped Image", cropped_image)

    # Perform bend curvature calculation
    # Convert the cropped image to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)
    # Apply edge detection
    edges = cv2.Canny(blurred, lower_bound, upper_bound)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # Display the edges
    cv2.imshow("Edges", edges)

    # Fit a polynomial to the contour points
    contour_points = contour[:, 0, :]
    x = contour_points[:, 0]
    y = contour_points[:, 1]
    z = np.polyfit(x, y, 5)
    p = np.poly1d(z)

    # create a continuous x range
    x_continuous = np.linspace(x.min(), x.max(), num=1000)
    y_continuous = p(x_continuous)

    # calculate curvature
    p_deriv1 = p.deriv(1)
    p_deriv2 = p.deriv(2)
    curvature = (
        np.abs(p_deriv2(x_continuous)) / (1 + p_deriv1(x_continuous) ** 2) ** 1.5 * 100
    )

    # draw contour points and curve
    for i in range(len(x_continuous) - 1):
        cv2.line(
            cropped_image,
            (int(x_continuous[i]), int(y_continuous[i])),
            (int(x_continuous[i + 1]), int(y_continuous[i + 1])),
            (0, 255, 0),
            2,
        )
        if i % 100 == 0:  # display curvature value every 100 points
            curv_text = f"{curvature[i]:.2f}"
            cv2.putText(
                cropped_image,
                curv_text,
                (int(x_continuous[i]), int(y_continuous[i])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

    # start point tangent
    draw_tangent(cropped_image, x_continuous[:10], y_continuous[:10])

    # end point tangent
    draw_tangent(cropped_image, x_continuous[-10:], y_continuous[-10:])

    # calculate the angle between the two tangents (can be calculated based on the fitted slopes)
    slope_start, _ = fit_line_to_points(x_continuous[:10], y_continuous[:10])
    slope_end, _ = fit_line_to_points(x_continuous[-10:], y_continuous[-10:])
    angle_start = np.arctan(slope_start)
    angle_end = np.arctan(slope_end)
    angle_between = abs(angle_start - angle_end)
    angle_between_degrees = np.degrees(angle_between)

    # mark the angle
    mid_x = int((x_continuous[0] + x_continuous[-1]) / 2)
    mid_y = int((y_continuous[0] + y_continuous[-1]) / 2)
    angle_text = f"{angle_between_degrees:.2f} deg"
    cv2.putText(
        cropped_image,
        angle_text,
        (mid_x, mid_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # display the result
    cv2.imshow("Rod Curvature", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cal_bend_cur_mannual():
    root = tk.Tk()
    root.withdraw()

    # Open file dialog to select image
    file_path = filedialog.askopenfilename()

    # Load image
    image = cv2.imread(file_path)

    # Perform image cropping based on selected region

    # Resize the image
    width = 500
    height = int(image.shape[0] * (width / image.shape[1]))
    resized_image = cv2.resize(image, (width, height))

    # Select ROI
    roi = cv2.selectROI("Select ROI", resized_image)

    # get the ROI in the original image (if you need the ROI in the original resolution)
    x, y, w, h = roi
    cropped_image = image[
        int(y * (image.shape[0] / height)) : int((y + h) * (image.shape[0] / height)),
        int(x * (image.shape[1] / width)) : int((x + w) * (image.shape[1] / width)),
    ]
    # Resize the cropped image to the same size as the original image
    cropped_image = cv2.resize(cropped_image, (width, height))

    # Display the cropped image
    cv2.imshow("Cropped Image", cropped_image)

    # mouse callback function
    def on_mouse(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Points", param)

    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", on_mouse, cropped_image)
    cv2.imshow("Select Points", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #  make sure at least 2 points are selected
    if len(points) < 2:
        print("Please select at least 2 points.")
        exit()

    # Fit a polynomial to the selected points
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)

    # create a continuous x range
    x_continuous = np.linspace(x.min(), x.max(), num=1000)
    y_continuous = p(x_continuous)

    # calculate curvature
    p_deriv1 = p.deriv(1)
    p_deriv2 = p.deriv(2)
    curvature = (
        np.abs(p_deriv2(x_continuous)) / (1 + p_deriv1(x_continuous) ** 2) ** 1.5 * 100
    )

    # draw contour points and curve
    for i in range(len(x_continuous) - 1):
        cv2.line(
            cropped_image,
            (int(x_continuous[i]), int(y_continuous[i])),
            (int(x_continuous[i + 1]), int(y_continuous[i + 1])),
            (0, 255, 0),
            2,
        )
        if i % 100 == 0:  # display curvature value every 100 points
            curv_text = f"{curvature[i]:.2f}"
            cv2.putText(
                cropped_image,
                curv_text,
                (int(x_continuous[i]), int(y_continuous[i])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

    # the start point tangent
    draw_tangent(cropped_image, x_continuous[:10], y_continuous[:10])

    # the end point tangent
    draw_tangent(cropped_image, x_continuous[-10:], y_continuous[-10:])

    # calculate the angle between the two tangents (can be calculated based on the fitted slopes)
    slope_start, _ = fit_line_to_points(x_continuous[:10], y_continuous[:10])
    slope_end, _ = fit_line_to_points(x_continuous[-10:], y_continuous[-10:])
    angle_start = np.arctan(slope_start)
    angle_end = np.arctan(slope_end)
    angle_between = abs(angle_start - angle_end)
    angle_between_degrees = np.degrees(angle_between)

    # mark the angle
    mid_x = int((x_continuous[0] + x_continuous[-1]) / 2)
    mid_y = int((y_continuous[0] + y_continuous[-1]) / 2)
    angle_text = f"{angle_between_degrees:.2f} deg"
    cv2.putText(
        cropped_image,
        angle_text,
        (mid_x, mid_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    #  display the result
    cv2.imshow("Rod Curvature", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cal_bend_cur_mannual()
    cal_bend_cur_auto(lower_bound=30, upper_bound=180)
