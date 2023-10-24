import os, sys
import cv2
import time
import pathlib
import click
import glob
from tqdm import tqdm
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import br2_vision
from br2_vision.utility.logging import config_logging, get_script_logger


@click.command()
@click.option(
    "-f",
    "--file",
    type=click.Path(exists=True),
    default=None,
    help="Calibration video path (or file).",
    multiple=True,
)
@click.option(
    "-s",
    "--skip-frame",
    type=int,
    default=30,
    help="Skip every n frames. (default: 30)",
)
@click.option(
    "-c",
    "--compression",
    type=int,
    default=6,
    help="PNG compression level 0-9. (default: 6)",
)
@click.option("-r", "--use-roi", is_flag=True, default=False, help="Use ROI")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose")
@click.option("-d", "--dry", is_flag=True, default=False, help="Dry run")
@click.option("-S", "--show", is_flag=True, default=False, help="Show frames")
def extract_frames(
    file, skip_frame, compression, use_roi: bool, verbose: bool, dry: bool, show: bool
):
    """
    Perform:
    ffmpeg -i input.mov -r 0.25 output_%04d.png
    ffmpeg -i input.mov -r 0.1 output_%04d.png
    """
    config = br2_vision.load_config()
    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    #
    raw_videos = []
    for f in file:
        if os.path.isdir(f):
            collections = glob.glob(
                f"{f}/*.{config['DEFAULT']['processing_video_extension']}",
                recursive=True,
            )
            raw_videos.extend(collections)
        else:
            raw_videos.append(f)

    for video_path in raw_videos:
        logger.info("Extracting frames from video file: {}".format(video_path))
        stime = time.time()
        if dry:
            continue
        cap = cv2.VideoCapture(video_path)
        video_path = pathlib.Path(video_path)
        directory = video_path.parent / video_path.stem
        if not directory.exists():
            directory.mkdir(parents=True)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        pbar = tqdm(total=total_frames)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        maxlen = 80
        prev_frames = deque(maxlen=maxlen)
        prev_similarity = 1.0
        similarity_threshold = 0.98
        similarities = []
        roi = None
        while cap.isOpened():
            ret, frame = cap.read()
            if ret and use_roi:
                if roi is None:
                    roi = cv2.selectROI(frame)
                    cv2.destroyAllWindows()
                frame = frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            if ret and (frame_count + 1) % skip_frame == 0:
                pbar.update(skip_frame)
                # if show:
                #    cv2.imshow('frame', frame)
                #    cv2.waitKey(1)

                filename = (
                    directory / "frame_{:05d}.png".format(frame_count)
                ).as_posix()
                similarity = cv2.matchTemplate(
                    prev_frames[0], frame, cv2.TM_CCOEFF_NORMED
                )[0][0]
                if (
                    prev_similarity > similarity_threshold
                    and similarity < similarity_threshold
                ):
                    cv2.imwrite(
                        filename,
                        prev_frames[0],
                        [cv2.IMWRITE_PNG_COMPRESSION, compression],
                    )
                    plt.axvline(
                        x=1.0 * (frame_count - maxlen) / skip_frame,
                        color="r",
                        linestyle="--",
                    )
                prev_similarity = similarity
                similarities.append(similarity)
            elif not ret:
                break
            frame_count += 1
            prev_frames.append(frame)
        cap.release()
        pbar.close()

        # DEBUG
        plt.plot(similarities)
        plt.xlabel("Frame")
        plt.ylabel("Similarity")
        plt.savefig(
            (video_path.parent / video_path.stem).as_posix() + "_similarity.png"
        )
        plt.close("all")

        logger.info("Elapsed time: {:.2f}s".format(time.time() - stime))
    logger.info("Done.")


if __name__ == "__main__":
    extract_frames()
