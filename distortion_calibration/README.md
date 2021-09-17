## undistort.py

The function 'undistort' is used to fix the fish-eye distortion of each camera. The calibration parameters are determined using multiple frames of checkerboard photos.
The calibration parameter must be pre-set to use the function.

## calibrate_grey.py

Use all the images in the directory (argument 0) to calibrate the camera
The calibartion parameters are printed, and the values can be used with ```undistort.py``` file.

## Reference
[Source 1](https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/)
