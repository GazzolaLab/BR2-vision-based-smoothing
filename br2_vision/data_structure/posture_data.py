import sys
import os
from dataclasses import dataclass
from typing import List, Tuple

import h5py
import numpy as np
from nptyping import Floating, NDArray, Shape
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from br2_vision.utility.convert_coordinate import get_center_and_normal
from br2_vision.utility.logging import get_script_logger

from .tracking_data import TrackingData
from .utils import raise_if_outside_context

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.cm as cm


def compute_positions_and_directors(
    path: str,
) -> tuple[
    NDArray[Shape["T, [x,y,z], N"], Floating], NDArray[Shape["T, 3, 3, N"], Floating]
]:
    """
    Compute positions and directors from the trajectory data
    """
    cross_section_center_position = []
    cross_section_director = []

    with TrackingData.load(path) as dataset:
        marker_positions = dataset.marker_positions
        marker_direction = marker_positions.marker_direction
        normal_direction = marker_positions.normal_direction

        for z_index in tqdm(range(len(marker_positions)), position=1):
            euler_coords = []
            data_collection = []
            for tag in marker_positions.tags:
                data = dataset.load_track(z_index, tag)
                if data is None:
                    continue
                euler_coords.append(marker_positions.get_position(z_index, tag, inplane=True))
                data_collection.append(data)
            euler_coords = np.array(euler_coords)

            positions = []
            directors = []
            for tidx in tqdm(range(len(data_collection[0])), position=0):
                P = np.array([data[tidx] for data in data_collection])  # shape: [N, 3]
                nan_index = np.isnan(P).any(axis=1)
                P = P[~nan_index]
                R = euler_coords[~nan_index]
                # Find Optimal Rotation Matrix
                p = P - P.mean(axis=1, keepdims=True)
                q = R - R.mean(axis=1, keepdims=True)
                H = p.T @ q
                u, s, vh = np.linalg.svd(H)
                d = np.linalg.det(u @ vh)
                Id = np.eye(3)
                Id[2, 2] = d
                Q = u @ Id @ vh
                if np.arccos(Q[:,2] @ marker_direction) > np.arccos(-Q[:,2] @ marker_direction):
                    Q = -Q
                # Find Optimal Translation Matrix
                r = ((R @ Q) - P).mean(axis=0)

                positions.append(r)
                directors.append(Q.T)  # Row matrix

                #A = LinearRegression().fit(R, P)
                #positions.append(A.predict([marker_positions.origin])[0])
                #directors.append(A.predict(marker_positions.Q.T).T)
            cross_section_center_position.append(np.array(positions))
            cross_section_director.append(np.array(directors))

    positions = np.stack(cross_section_center_position, axis=-1)
    directors = np.stack(cross_section_director, axis=-1)

    print("Posture Interpolation")
    print(f"{positions.shape=}")
    print(f"{directors.shape=}")

    return positions, directors


class PostureData:
    """
    Data structure for postures
    """

    def __init__(self, path):
        self.path = path
        self.logger = get_script_logger(os.path.basename(__file__))

        self._inside_context = False

        self._positions: NDArray[Shape["T, [x,y,z], N"], Floating] = None
        self._positions_key: str = "/posture/positions"
        self._directors: NDArray[Shape["T, 3, 3, N"], Floating] = None
        self._directors_key: str = "/posture/directors"
        self._time: NDArray[Shape["T"], Floating] = None
        self._time_key: str = "/dlt-track/timestamps"

    @raise_if_outside_context
    def get_time(self):
        return self._time

    @raise_if_outside_context
    def get_cross_section_center_position(self):
        return self._positions

    @raise_if_outside_context
    def get_cross_section_director(self):
        return self._directors

    @raise_if_outside_context
    def save_positions_and_directors(self):
        with h5py.File(self.path, "a") as h5f:
            position_dataset = h5f.require_dataset(
                self._positions_key,
                self._positions.shape,
                dtype=np.float64,
            )
            position_dataset[...] = self._positions
            position_dataset.attrs["unit"] = "m"

            director_dataset = h5f.require_dataset(
                self._directors_key,
                self._directors.shape,
                dtype=np.float64,
            )
            director_dataset[...] = self._directors
            director_dataset.attrs["unit"] = "m"

        print("Posture saved.")

    @raise_if_outside_context
    def load_positions_and_directors(self):
        """
        Load position and directors if exists.
        If not, compute them from the trajectory data.
        """

        flag = False
        with h5py.File(self.path, "r") as h5f:
            if self._positions_key in h5f:
                self._positions = np.array(h5f[self._positions_key], dtype=np.float64)
            else:
                flag = True

            if self._directors_key in h5f:
                self._directors = np.array(h5f[self._directors_key], dtype=np.float64)
            else:
                flag = True

            self._time = np.array(h5f[self._time_key], dtype=np.float64)

        if flag:
            self.logger.info("Computing positions and directors...")
            self.compute_posture()
            self.logger.info("Computing positions and directors Done.")

    @raise_if_outside_context
    def compute_posture(self):
        self._positions, self._directors = compute_positions_and_directors(
            self.path
        )

    def __enter__(self):
        """
        If file at self.path does not exist, create one.
        """
        self._inside_context = True
        assert os.path.exists(self.path), f"File does not exist {self.path}."
        self.load_positions_and_directors()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Save posture on the existing file
        """
        self.save_positions_and_directors()
        self._inside_context = False

    def debug_plot(self, path):
        self.plot_pose(path)
        self.render_pose_video(path)

    def plot_pose(self, path):
        # Configuration
        dpi = 300
        #color_scheme = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        positions = self._positions
        directors = self._directors
        n_plane = positions.shape[-1]

        # Position
        fig, axes = plt.subplots(3,1,figsize=(10,12), sharex=True)
        axes[0].plot(positions[:,0,:], '.')
        axes[0].legend([str(i) for i in range(n_plane)])
        axes[0].grid(True)
        axes[0].set_ylabel('x')
        axes[1].plot(positions[:,1,:], '.')
        axes[1].legend([str(i) for i in range(n_plane)])
        axes[1].grid(True)
        axes[1].set_ylabel('y')
        axes[2].plot(positions[:,2,:], '.')
        axes[2].legend([str(i) for i in range(n_plane)])
        axes[2].grid(True)
        axes[2].set_ylabel('z')
        axes[2].set_xlabel('frame')
        plt.savefig(os.path.join(path, "xyz.png"), dpi=dpi)
        plt.close('all')

        # Directors
        fig, axes = plt.subplots(3,3,figsize=(10,12), sharex=True, sharey=True)
        for i in range(3):
            for j in range(3):
                axes[i,j].plot
                axes[i,j].plot(directors[:,i,j,:], '.')
                axes[i,j].legend([str(i) for i in range(n_plane)])
                axes[i,j].grid(True)
        plt.savefig(os.path.join(path, "Q.png"), dpi=dpi)
        plt.close('all')

    def render_pose_video(self, path):
        # Configuration
        video_name = os.path.join(path, "interpolated_poses.mp4")
        dpi = 300
        step = 1
        fps = 60
        color_scheme = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        quiver_length = 0.010

        positions = self._positions
        directors = self._directors
        print(f"{positions.shape=}")
        print(f"{directors.shape=}")

        n_visualized_plane = positions.shape[-1]
        timesteps = positions.shape[0]

        # Prepare Matplotlib axes
        fig = plt.figure(1, figsize=(10, 8))
        ax = plt.axes(projection="3d")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        #ax.set_xlim((-0.13, 0.13))
        #ax.set_ylim((-0.05, 0.3))
        #ax.set_zlim((-0.13, 0.13))
        xrange = positions[:, 0, :].max() - positions[:, 0, :].min()
        yrange = positions[:, 1, :].max() - positions[:, 1, :].min()
        zrange = positions[:, 2, :].max() - positions[:, 2, :].min()
        ax.set_xlim((positions[:, 0, :].min()-xrange*.5, positions[:, 0, :].max()+xrange*.5))
        ax.set_ylim((positions[:, 1, :].min()-yrange*.5, positions[:, 1, :].max()+yrange*.5))
        ax.set_zlim((positions[:, 2, :].min()-zrange*.5, positions[:, 2, :].max()+zrange*.5))

        #ax.view_init(elev=-40, azim=-90)

        # Write Video
        FFMpegWriter = animation.writers["ffmpeg"]
        writer = FFMpegWriter(fps=fps)
        with writer.saving(fig, video_name, dpi):
            time_idx = 0
            quiver_axes = [[] for _ in range(n_visualized_plane)]
            texts = []
            # _rot = [] # Delete later
            for idx in range(n_visualized_plane):
               position = positions[time_idx,:,idx]
               director = directors[time_idx,:,:,idx]
               #_rot.append(np.matmul(directors[time_idx,:,:,0], director.T)) # Delete later
               #director = np.matmul(_rot[idx], director) # Delete later
               for i in range(3):
                   quiver_axes[idx].append(ax.quiver(*position, *director[i], color=color_scheme[i])) #idx%10]))
               texts.append(ax.text2D(0, 0, f"plane {idx}"))
            # writer.grab_frame()

            # Create a custom legend
            handles = [plt.Line2D([0], [0], color=color_scheme[i], lw=4) for i in range(3)]
            labels = ['n', 'b', 't']

            # Add the legend to the plot
            ax.legend(handles, labels, loc='best')

            ax.set_aspect("equal")
            #ax.set_aspect("auto")
            position_sc = ax.scatter([], [], [], color="red", s=10)
            for time_idx in tqdm(range(0, timesteps, int(step))):
                position_sc._offsets3d = (
                    positions[time_idx, 0, :],
                    positions[time_idx, 1, :],
                    positions[time_idx, 2, :],
                )
                for idx in range(n_visualized_plane):
                    position = positions[time_idx,:,idx]
                    director = directors[time_idx,:,:,idx]
                    #director = np.matmul(_rot[idx], director) # Delete later
                    director *= quiver_length
                    for i in range(3):
                        segs = [[position.tolist(), (position+director[i,:]).tolist()]]
                        quiver_axes[idx][i].set_segments(segs)
                    _x, _y, _ = proj3d.proj_transform(*position, ax.get_proj())
                    texts[idx].set_position((_x, _y))
                writer.grab_frame()
        plt.close(plt.gcf())


