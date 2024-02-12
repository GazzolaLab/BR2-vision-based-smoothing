import cv2


def select_roi(video_path):
    """
    Select roi using cv2.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        return None

    r = cv2.selectROI(frame)
    cv2.destroyAllWindows()
    return r
