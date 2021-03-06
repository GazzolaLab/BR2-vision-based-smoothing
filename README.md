# A physics-informed, vision-based method to reconstruct all deformation modes in slender bodies

[Paper (arXiv)](https://arxiv.org/abs/2109.08372) - In Review (RA-L with ICRA 2022) 

[Demo Data](https://uofi.box.com/s/7wjf2wrtq6ykn5km7umng4mf6reme3sq)

## Requirements

All the tools are developed using python 3 and openCV.
The open-source package numpy/numba is used to process the data.
The remaining dependencies are listed in [requirements.txt](requirements.txt) file.
To visualize the data, we used `ffmpeg` tool to render video.

- [PyElastica 0.1.0](https://github.com/GazzolaLab/PyElastica)
- python 3.6+
- numpy/numba
- Matplotlib
- OpenCV 4.5
- ffmpeg
- PyQt (optional, not used in reconstruction)

_(The PyQt may not operate in some OS environment)_

## Publication

Kim, Chang, Shih, Uppalapati, Halder, Krishnan, Mehta and Gazzola <strong>A physics-informed , vision-based method to reconstruct all deformation modes in slender bodies</strong>, IEEE Robotics and Automation Letters (In Review)

```
@article{Kim2021,
abstract = {This paper is concerned with the problem of esti- mating (interpolating and smoothing) the shape (pose and the six modes of deformation) of a slender flexible body from multiple camera measurements. This problem is important in both biol- ogy, where slender, soft, and elastic structures are ubiquitously encountered across species, and in engineering, particularly in the area of soft robotics. The proposed mathematical formulation for shape estimation is physics-informed, based on the use of the special Cosserat rod theory whose equations encode slender body mechanics in the presence of bending, shearing, twisting and stretching. The approach is used to derive numerical algorithms which are experimentally demonstrated for fiber reinforced and cable-driven soft robot arms. These experimental demonstrations show that the methodology is accurate (<5 mm error, three times less than the arm diameter) and robust to noise and uncertainties. CONTINUUM},
author = {Kim, Seung Hyun and Chang, Heng-Sheng and Shih, Chia-Hsien and Uppalapati, Naveen Kumar and Halder, Udit},
file = {:Users/skim0119/Documents/Mendeley Desktop/2021 - Kim et al. - A physics-informed , vision-based method to reconstruct all deformation modes in slender bodies.pdf:pdf},
title = {{A physics-informed , vision-based method to reconstruct all deformation modes in slender bodies}},
year = {2021}
}
```

## How To Use

### Path Configuration

All data paths and output paths can be changed in `config.py` file.

### Reconstruction (Smoothing)

1. Check that the posture data file is inside the folder named `data` and the data file name is `<Keyword>.npz`.
* Default Sample `<Keyword>` Options: `bend` / `twist` / `mix` / `cable`.
* Note that if you have your own data, please properly modify the `delta_s_position` variable in the `main()` function of the `run_smoothing.py` file.
* The `delta_s_position` parameter decides the distance between each marker in the rest state of the soft arm.

2. Run the smoothing algorithm

``` bash
python run_smoothing.py --problem <Keyword>
```

3. Once the algorithm is completed, the processed data will then be stored in the `result_data` folder and is named as `<Keyword>.pickle`.

4. To visualize the result, run
``` bash
python visualization.py --problem <Keyword>
```
This will create a visualization video named `<Keyword>.mov` and a folder named `frames` with all frame results in it.

### Calibration Steps

1. Select DLT calibration points

Select calibration point for all calibration frames in the directory.
Save the points in camera (each) coordinate and lab-coordinate.

```bash
python run_calibration_points.py
```

The red mark on reference point indicates 'locked' status.
Using locked points, we estimate the remaining reference locations with inverse DLT.
Locked points does not move after the interpolation; only blue points are re-evaluated and re-located.
Each points have unique label that is used to interpolate the true coordinate.

- Left Click: Select point
    - Control-Left Click: Delete point
- Right Click: Lock point
    - Control-Right Click: Lock all points
- Key 'D' or 'd': Delete last coordinate
- Key 'b': Label points
- Key 'p': Interpolate using planar(2D) DLT (At least 4 locked points are required)
- Key 'P': Use Harry's corner detection.
- Key 'o': Use 3D DLT (from other reference frame images)
' Key 's': Save
- 'Enter,' 'Space,' 'ESC': Complete

2. Calibration

```bash
python run_calibration.py
```

Read all 2D calibration points and determines the camera parameters.
Calibration configuration is saved.

3. Check Calibration Error

```bash
python calibration_error.py
```

### Optical Flow: Data Point Tracking

1. Initial Reference Point

```bash
python add_initial_flow_point.py -h
python add_initial_flow_point.py --camid <camera id> --runid <run id>
```

The script is used to select multiple initial flow position and to label them.
The desire window of flow can be determined using the optional argument `start_frame` and `end_frame` (default is from first to last frame).
Multiple flow position can be selected.

- Left Click: Click the pixel to label
- Right Click: Zoom-in
    - Left Click: Reselect the point
    - Right Click: Return to original point
    - Key 'a': accept the selection
    - Key 'd': decline the selection
- Key 'p': Print the selection
- Key 'd': Delete the last seletion
- Key 'c': Complete

2. Optical Flow

```bash
python run_optical_flow.py --camid <camera id> --runid <run id>
```

Run optical flow given the initial poiition selected from `add_initial_flow_point.py`.
The script export the result in a video(mp4) format.

3. DLT Process

```bash
python process_dlt.py --runid 1 
```

Convert each tracked point to 3d point using DLT, and save in 'runRUNID-position.npz' file.

4. Interpolate Cross-Section Data

```bash
python data_augment_cross_section.py
```

### Overlay Result

```bash
python overlay_smoothing.py
```

Plot simulated data points on camera space.
Careful choice of axis is required

### Utility Scripts

- dlt.py
    - Contain all DLT related methods.

### Video Pre-processing

Following preprocessing scripts are included.

- undistort_rotate.py: undistort and rotate video
- undistort_rotate_calibration.py: undistort and rotate calibration frame 
- trim_video_intervals.py: detect led and trim video

