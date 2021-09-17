import cv2
import numpy as np
import os
import glob

# Calibrated: 06/22/2021
#DIM = (1920, 1080)
#K = np.array([[1027.742399462262, 0.0, 960.8670538113306], [0.0, 982.6130074786815, 419.8587142488236], [0.0, 0.0, 1.0]])
#D = np.array([[0.316614958125011], [-0.5886345366993294], [1.1914884993213208], [-0.7797830401841401]])

# Calibrated:  2021-06-24 17:34:43.386226
# Found 39 valid images for calibration
DIM=(1920, 1080)
K=np.array([[1099.5325618359605, 0.0, 964.8349560375052], [0.0, 1081.2293744495942, 539.3268284303385], [0.0, 0.0, 1.0]])
D=np.array([[0.2438938497997481], [-0.2522190386594705], [0.5028241919789112], [-0.4013399826024004]])

def undistort(image):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_image
