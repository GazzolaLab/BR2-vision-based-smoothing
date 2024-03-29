import cv2

from .transformation import scale_image


def select_roi(video_path, scale=None):
    """
    Select roi using cv2.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        return None
    cap.release()

    cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)

    if isinstance(scale, float):
        frame = scale_image(frame, scale)
        r = cv2.selectROI("frame", frame, False, False)
        # unscale roi
        r = (int(r[0] / scale), int(r[1] / scale), int(r[2] / scale), int(r[3] / scale))
    else:
        r = cv2.selectROI("frame", frame, False, False)
    cv2.destroyAllWindows()
    return r


def crop_roi(image, roi):
    y, x, disp_h, disp_w = roi

    # Region of interest display
    padded_img = cv2.copyMakeBorder(
        image,
        disp_h // 2,
        disp_h // 2,
        disp_w // 2,
        disp_w // 2,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    cropped_img = padded_img[y : y + disp_h, x : x + disp_w]
    return cropped_img
