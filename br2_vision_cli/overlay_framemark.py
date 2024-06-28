import os
import pathlib
import subprocess
import sys

import click


def extract_video_with_framemark(input_path: str, output_path: str):
    # mark frame number and time
    command = ["ffmpeg", "-y"]
    command.extend(["-i", f"{input_path}"])
    command.extend(
        [
            "-vf",
            "drawtext=text='Frame %{n}, Time\: %{pts\:gmtime\:0\:%H\\\:%M\\\:%S}':fontsize=30:fontcolor=white:borderw=2:bordercolor=black:x=10:y=h-th-10",
        ]
    )
    command.extend(["-c:a", "copy"])
    command.extend([f"{output_path}"])
    command = " ".join(command)
    print("running : ", command)

    sts = subprocess.Popen(command, shell=True).wait()
    return sts


@click.command()
@click.option(
    "-f",
    "--filepath",
    type=str,
    help="video file path",
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="output file path",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def process(filepath, output, verbose: bool, dry: bool):
    """
    Overlay frame number and time on the video
    """
    assert os.path.exists(filepath)
    extract_video_with_framemark(filepath, output)


if __name__ == "__main__":
    process()
