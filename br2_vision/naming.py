import os
import glob

# DATA PATH
PATH                             = 'data'

# VIDEO DATA PATH
RAW_VIDEO_PATH                   = os.path.join(PATH, 'raw', 'cam-{}-row.MOV') # camera id
RAW_FOOTAGE_VIDEO_PATH           = os.path.join(PATH, 'raw', 'cam-{}-footage-{}.mp4') # camera id, run id
RAW_FOOTAGE_VIDEO_PATH_WILD      = os.path.join(PATH, 'raw', 'cam-{}-footage-*.mp4') # camera id, run id

# CALIBRATION PATH
CALIBRATION_PATH                 = os.path.join(PATH, 'calibration')
CALIBRATION_IMAGE_COLLECTION     = os.path.join(PATH, 'calibration', 'cam-{}-*.png')
CALIBRATION_REF_POINTS_PATH      = os.path.join(PATH, 'calibration', 'calibration_points.npz')
CALIBRATION_REF_POINT_SAVE       = os.path.join(PATH, 'calibration', 'cam-{}-calibration-{}-refpoints.npz')
CALIBRATION_REF_POINT_SAVE_WILD  = os.path.join(PATH, 'calibration', 'cam-{}-calibration-*-refpoints.npz')
CALIBRATION_DLT_PATH             = os.path.join(PATH, 'calibration', 'cam-{}-calibration-{}-dlt.npz')
CALIBRATION_VIEW_PATH            = os.path.join(PATH, 'calibration', 'cam-{}-calibration-{}-view.npz')

# POSTPROCESSING PATH
POSTPROCESSING_PATH              = os.path.join(PATH, 'postprocess')
PREPROCESSED_FOOTAGE_VIDEO_PATH  = os.path.join(PATH, 'postprocess', 'cam-{}-footage-{}.mp4') # camera id, run id
PREPROCESSED_TRACKING_VIDEO_PATH = os.path.join(PATH, 'postprocess', 'cam-{}-footage-{}-tracking.mp4') # camera id, run id
PREPROCESSED_POSITION_PATH       = os.path.join(PATH, 'postprocess', 'run-{}-position.npz')
TRACKING_FILE                    = os.path.join(PATH, 'postprocess', 'cam-{}-footage-{}-ps.npz')
OVERLAY_SMOOTHING_VIDEO_PATH     = os.path.join(PATH, 'postprocess', 'cam-{}-footage-{}-smoothing-overlay.mp4')


