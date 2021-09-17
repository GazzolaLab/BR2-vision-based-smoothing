import os
import glob

# DATA PATH
PATH = 'data'

# VIDEO DATA PATH
RAW_VIDEO_PATH = os.path.join(PATH, 'raw', 'cam-{}-row.MOV') # camera id
RAW_FOOTAGE_VIDEO_PATH = os.path.join(PATH, 'raw', 'cam-{}-footage-{}.mp4') # camera id, run id
RAW_FOOTAGE_VIDEO_PATH_WILD = os.path.join(PATH, 'raw', 'cam-{}-footage-*.mp4') # camera id, run id
PREPROCESSED_FOOTAGE_VIDEO_PATH = os.path.join(PATH, 'postprocess', 'cam-{}-footage-{}.mp4') # camera id, run id

# CALIBRATION PATH
CALIBRATION_PATH = os.path.join(PATH, 'calibration')
CALIBRATION_IMAGE_COLLECTION = os.path.join(PATH, 'calibration', 'cam-{}-*.png'.)


