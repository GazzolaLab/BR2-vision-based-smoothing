import cv2

def get_video_frame_count(video_path):
    """
    Return frame-count of the video
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count
