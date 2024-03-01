import cv2
import sys
import pathlib
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import os
import glob

import click

import br2_vision
from br2_vision.utility.logging import config_logging, get_script_logger


def get_led_state(frame, roi, led_threshold):
    """ """
    # Crop Image
    crop = frame[int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])]

    # Determine State
    led_threshold = np.asarray(led_threshold)
    average_color = crop.mean(axis=0).mean(axis=0)

    # LED Threshold
    new_state = np.linalg.norm(average_color) > np.linalg.norm(led_threshold)
    return new_state, average_color


def export(input_path, output_path, start_stamp, end_stamp):
    # TODO: dont use os.system, use subsystem
    command = f"ffmpeg -y -i {input_path} -ss {start_stamp} \
            -to {end_stamp} {output_path}"
    os.system(command)


@click.command()
@click.option(
    "-t",
    "--tag",
    type=str,
    help="Experiment tag. Path ./tag should exist.",
)
@click.option(
    "-c", "--cam-id", type=int, help="Camera index given in file.", multiple=True
)
@click.option(
    "-f",
    "--fps",
    type=int,
    default=60,
    help="Frames per seconds (default=60). Make sure it matches the orignal framerate of the video.",
)
@click.option("-r", "--run-id", type=int, default=0, help="Run index given in file", show_default=True)
@click.option(
    "-tr",
    "--trailing-frames",
    type=int,
    default=0,
    help="Number of trailing frames after the LED status is turned off. (default=0)",
)
@click.option(
    "--led-threshold",
    type=(int, int, int),
    default=(50, 50, 150),
    help='RGB threshold of the LED: greater value will be considered as "on"',
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def process(
    tag, cam_id, fps, run_id, trailing_frames, led_threshold, verbose: bool, dry: bool
):
    """
    Trimming process. Script asks for ROI and trim the video.

    The video includes LED light that indicates the active status of the experiment.
    The script asks for a region-of-interest (roi), which selects where the LED is
    visible, and trim the video for each interval that LED light is on.

    Parameters
    ----------
    tag : pathlib.Path
        Experiment tag. The raw data are expected to be stored in the path ./tag
    cam_id : int
        Camera index given in file (multiple).
    fps : int
        Frames per seconds (default=60)
    run_id : int
        Run index given in file
    trailing_frames : int
        Number of trailing frames after the LED status is turned off. (default=0)
    led_threshold : tuple(int,int,int)
        RGB threshold of the LED: greater value will be considered as 'on'
    """
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    # Utility lambdas
    frame2timestr = lambda frame: str(frame / fps)

    # Path Configuration
    working_dir = config["PATHS"]["postprocessing_path"].format(tag)
    os.makedirs(working_dir, exist_ok=True)
    video_path = config["PATHS"]["experiment_video_path"].format(
        tag, "{}"
    )  # (cam_id)
    output_path = config["PATHS"]["footage_video_path"].format(
        tag, "{}", "{}"
    )  # (cam_id, run_id)

    # Select LED regions for all cameras
    rois = []
    for i in cam_id:
        video_path_ = video_path.format(i)
        logger.info(f"Processing camera {i}: {video_path_}")
        capture = cv2.VideoCapture(video_path_)
        ret, frame = capture.read()
        if not ret:
            continue
        else:
            logger.info(f"Processing camera {i}: frame shape {frame.shape}")
        r = cv2.selectROI("select roi", frame)
        logger.info(f"ROI cam-{i} is selected: {r}")
        rois.append(r)
        capture.release()
    cv2.waitKey(500)  # Ensure all windows close
    cv2.destroyAllWindows()

    # Iterate all cameras
    captures = [cv2.VideoCapture(video_path.format(i)) for i in cam_id]
    captures_status = [True for i in cam_id]

    # Iterate Video and export
    current_state = False
    frame_count = 0
    num_frames = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
    total_frame = min(num_frames)
    logger.info(f"Total frame: {total_frame}")
    pbar = tqdm(total=total_frame)
    state_list = []
    colors_list = [[] for _ in cam_id]
    assert (
        total_frame > 0
    ), f"Minimum frame for some video is zero. Check if all the input video exist."
    while frame_count < total_frame:
        states = []
        for i in range(len(cam_id)):
            ret, frame = captures[i].read()
            captures_status[i] = ret
            if ret and (rois[i][2] > 0 and rois[i][3] > 0):
                _state, ave_color = get_led_state(frame, rois[i], led_threshold)
                states.append(_state)
                colors_list[i].append(ave_color)
        if not all(captures_status):
            break
        frame_count += 1
        pbar.update(1)

        new_state = any(states)
        state_list.append(new_state)
        if dry:
            continue

        if new_state and not current_state:
            # Mark the start-frame
            start_frame = frame_count
            current_state = True
        elif not new_state and current_state:
            # Mark the end-frame
            end_frame = frame_count + trailing_frames
            # Trim
            start_stamp = frame2timestr(start_frame)
            end_stamp = frame2timestr(end_frame)
            current_state = False
            logger.info(f"Trimming: {start_stamp=} to {end_stamp=}")
            for i in cam_id:
                export(
                    input_path=video_path.format(i),
                    output_path=output_path.format(i, run_id),
                    start_stamp=start_stamp,
                    end_stamp=end_stamp,
                )

            run_id += 1

    # Release all captures
    for capture in captures:
        capture.release()
    pbar.close()

    # Plot LED state
    plt.plot(state_list)
    plt.xlabel("Frame")
    plt.ylabel("LED State")
    plt.savefig(working_dir + "/led_state.png")
    plt.close()

    # Plot LED color
    for i in range(len(cam_id)):
        if rois[i][2] == 0 or rois[i][3] == 0:
            continue

        colors = np.asarray(colors_list[i])
        plt.plot(colors[:, 0], color="blue", label="B")
        plt.plot(colors[:, 1], color="green", label="G")
        plt.plot(colors[:, 2], color="red", label="R")
        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("LED Color")
        plt.savefig(working_dir + f"/led_color_cam{i}.png")
        plt.close()


if __name__ == "__main__":
    process()
