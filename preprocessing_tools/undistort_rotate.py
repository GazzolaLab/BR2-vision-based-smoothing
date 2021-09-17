import cv2
import glob
import os, sys
import argparse

from undistort_tools.undistort import undistort

from config import *

"""
Undistort and rotate the video.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--camid', type=int, default=1, help='Camera Index')
parser.add_argument('--rotate', type=str, default=None, help='Rotation in cv2 (ex:ROTATE_90_CLOCKWISE)')
args = parser.parse_args()

camid = args.camid
if args.rotate is not None:
    cv2_rotation = getattr(cv2, args.rotate, None)
else:
    cv2_rotation = None

raw_video = glob.glob(RAW_FOOTAGE_VIDEO_PATH_WILD.format(camid), recursive=True)

for video_path in raw_video:
    basename = os.path.basename(video_path)[:-4]
    save_path = os.path.join(PATH, 'postprocess', basename+'.mp4')
    print(f'{save_path=}')

    # Create an object to read 
    video = cv2.VideoCapture(video_path)
       
    # We need to check if camera
    # is opened previously or not
    if (video.isOpened() == False): 
        print("Error reading video file {}".format(video_path))
        continue
      
    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
       
    size = (frame_height, frame_width) # Make sure the size is upright
       
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter(save_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             60, size)
        
    print('writing video...')
    while(True):
        ret, frame = video.read()
        if not ret:
            break

        # Undistort
        frame = undistort(frame)

        # Rotate
        if cv2_rotation != None:
            frame = cv2.rotate(frame, cv2_rotation)

        # Write video
        result.write(frame)
      
    # When everything done, release 
    # the video capture and video 
    # write objects
    video.release()
    result.release()
       
    print("The video was successfully saved - {}".format(save_path))
