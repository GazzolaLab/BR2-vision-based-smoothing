# A physics-informed, vision-based method to reconstruct all deformation modes in slender bodies

[Paper (arXiv)]() - In Review (RA-L with ICRA 2022) 
[Sample Data](https://uofi.box.com/s/7wjf2wrtq6ykn5km7umng4mf6reme3sq)

## How To Use

All the tools are developed using python 3 and openCV.
The open-source package numpy/numba is used to process the data.
The remaining dependencies are listed in [requirements.txt](requirements.txt) file.
To visualize the data, we used ```ffmpeg``` tool to render video.

- python 3.6+
- numpy/numba
- Matplotlib
- ffmpeg
= PyQt (optional)

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

## Instruction
### Video Pre-processing

- Defish: 718
    - TODO: Find better way to defish the captured frames
- Calibration frames (control frames)

- undistort_rotate.py
    - undistort and rotate video (take time)
- undistort_rotate_calibration.py
    - undistort and rotate calibration frame 
- trim_led.py (TODO: do trim before undistort_rotate operation)
    detect led and trim video (take time)
- undistort_rotate_calibration.py
    - undistort and rotate images


### Calibration Steps

Prepare all calibration video.
First, try to extract calibration frames.

calibration_frame_extract.py
undistort_rotate_calibrations_batch.sh

1. DLT calibration points

```bash
python calibration_point_selection.py
```

Select calibration point from control frames.
Save the points in camera (each) coordinate and lab-coordinate.

2. Calibration

```bash
python calibration.py --camid 1
```

Read calibration points and output L and R matrix in 'calibration.npz' file.
Calibration configuration is saved in this step.

### Data Point Tracking

1. Initial Reference Point

```bash
python trackinge_point_selection.py
```

Select the initial reference points and save in 'cam-CAMID-run-p0.npz' file.

2. Optical Flow

```bash
python optical_flow.py --runid 1 --camid 1
python optical_flow.py --runid 1 --camid 2
```

Track the point on camera space and save in 'cam-CAMID-runRUNID-ps.npz' file.
If OUTPUT_RENDERING is set to true, save tracking video 'cam-CAMID-runRUNID-tracking.mp4' video.

Note, the script must run for each video.

3. DLT Process

```bash
python process_dlt.py --runid 1 
```

Convert each tracked point to 3d point using DLT, and save in 'runRUNID-position.npz' file.

### Utilities

- Color Clustering

```bash
python color_cluster.py --path <image_path.png> -N 10
```

Create cluster of color in given image.

### Debug

- Plot axis on calibration image.

```bash
python project_axis.py --camid 1
python project_axis.py --camid 2
```

- Inverse dlt

```bash
python process_dlt_inverse.py --runid 1
```

Plot simulated data points on camera space.
Careful choice of axis is required


### Utility Scripts

- dlt.py
    - Contain all DLT related methods.

### Smoothing

1. Check that the posture data file is inside the folder named ```data``` and the data file name is ```<Keyword>.npz```.
* Default Sample ```<Keyword>``` Options: ```bend``` / ```twist``` / ```mix``` / ```cable```.
* Note that if you have your own data, please properly modify the ```delta_s_position``` variable in the ```main()``` function of the ```run_smoothing.py``` file.
* The ```delta_s_position``` parameter decides the distance between each marker in the rest state of the soft arm.

2. Run the smoothing algorithm
``` bash
python run_smoothing.py --problem <Keyword>
```

3. Once the algorithm is completed, the processed data will then be stored in the ```result_data``` folder and is named as ```<Keyword>.pickle```.

4. To visualize the result, run
``` bash
python visualization.py --problem <Keyword>
```
This will create a visualization video named ```<Keyword>.mov``` and a folder named ```frames``` with all frame results in it.

