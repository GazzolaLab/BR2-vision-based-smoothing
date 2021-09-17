"""
Created on Jan. 08, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import os, sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

def include_parent_folders(parent_folders):
    for parent_folder in parent_folders:
        path = os.path.abspath(__file__)
        for directory in path.split("/")[::-1]:
            if directory == parent_folder:
                break
            path = os.path.dirname(path)
        sys.path.append(path)

include_parent_folders(
    parent_folders=[
        "elastica-python",
        "Smoothing",
    ]
)

from frames.frame import Frame
from rod_tools import sigma_to_shear, kappa_to_curvature, calculate_dilatation

class RodFrame(Frame):
    def __init__(self, figure_name, folder_name, fig_dict, gs_dict, **kwargs):
        Frame.__init__(
            self,
            figure_name=figure_name,
            folder_name=folder_name,
            fig_dict=fig_dict
        )
        self.gs_dict = gs_dict

        self.fontsize = kwargs.get("fontsize", 15)

        self.rod_color = kwargs.get("rod_color", mcolors.BASE_COLORS['m'])

        self.reference_length = 1
        self.reference_configuration_flag = False

        self.n_elems = kwargs.get("n_elems", 100)
        self.s = np.linspace(0, 1, self.n_elems+1)
        self.s_external_force = self.s.copy()
        self.s_external_couple = (self.s[:-1] + self.s[1:])/2
        self.s_shear = (self.s[:-1] + self.s[1:])/2
        self.s_curvature = self.s[1:-1].copy()

        self.ax3d_flag = kwargs.get("ax3d_flag", False)
        if self.ax3d_flag:
            self.plot_rod = self.plot_rod3d
        else:
            self.plot_rod = self.plot_rod2d

    def reset(self,):
        Frame.reset(self,)
        self.gs = gridspec.GridSpec(
            figure=self.fig,
            **self.gs_dict
        )

        self.axes_shear = []
        self.axes_curvature = []
        for index_i in range(3):
            self.axes_curvature.append(
                self.fig.add_subplot(self.gs[index_i, 5], xlim=[-0.1, 1.1])
            )
            self.axes_shear.append(
                self.fig.add_subplot(self.gs[index_i, 4], xlim=[-0.1, 1.1])
            )

        if self.ax3d_flag:
            self.ax_rod = self.fig.add_subplot(self.gs[0:3, 0:3], projection='3d')
        else:
            self.ax_rod = self.fig.add_subplot(self.gs[0:3, 0:3])
        
        if self.reference_configuration_flag:
            self.plot_ref_configuration()
        
        
    def set_ref_configuration(self, position, shear, kappa, reference_length):
        self.reference_position = position.copy()
        
        _, voronoi_dilatation = calculate_dilatation(shear)

        self.reference_shear = shear.copy()
        self.reference_curvature = kappa_to_curvature(kappa, voronoi_dilatation)
        
        self.reference_length = reference_length.sum()
        self.n_elems = reference_length.shape[0]
        self.s = np.linspace(0, 1, self.n_elems+1)
        self.s_external_force = self.s.copy()
        self.s_external_couple = (self.s[:-1] + self.s[1:])/2
        self.s_shear = (self.s[:-1] + self.s[1:])/2
        self.s_curvature = self.s[1:-1].copy()
        self.reference_configuration_flag = True

    def plot_ref_configuration(self,):
        p0 = self.reference_position / self.reference_length
        self.ax_rod.plot(p0[0], p0[1], p0[2], color="grey", linestyle="--",)

        for index_i in range(3):
            self.axes_shear[index_i].plot(
                self.s_shear,
                self.reference_shear[index_i],
                color="grey",
                linestyle="--"
            )
            self.axes_curvature[index_i].plot(
                self.s_curvature,
                self.reference_curvature[index_i],
                color="grey",
                linestyle="--"
            )

    def plot_rod2d(self, position, director, radius, color=None):
        p_center = position / self.reference_length
        p = (position[:, :-1] + position[:, 1:])/2
        p_up = (p + director[1, :, :] * radius) / self.reference_length
        p_down = (p - director[1, :, :] * radius) / self.reference_length
        p_left = (p + director[0, :, :] * radius) / self.reference_length
        p_right = (p - director[0, :, :] * radius) / self.reference_length

        self.ax_rod.plot(p_up[0], p_up[1], color=self.rod_color if color is None else color)
        self.ax_rod.plot(p_down[0], p_down[1], color=self.rod_color if color is None else color)
        self.ax_rod.plot(p_left[0], p_left[1], alpha=0.2, color=self.rod_color if color is None else color)
        self.ax_rod.plot(p_right[0], p_right[1], alpha=0.2, color=self.rod_color if color is None else color)
        self.ax_rod.plot(
            [p_up[0, -1], p_center[0, -1], p_down[0, -1]],
            [p_up[1, -1], p_center[1, -1], p_down[1, -1]],
            color=self.rod_color if color is None else color,
            label='sim'
        )
        self.ax_rod.plot(
            [p_left[0, -1], p_center[0, -1], p_right[0, -1]],
            [p_left[1, -1], p_center[1, -1], p_right[1, -1]],
            alpha=0.2,
            color=self.rod_color if color is None else color
        )
        return self.ax_rod

    def plot_rod3d(self, position, director, radius, color=None):
        p_center = position / self.reference_length
        p = (position[:, :-1] + position[:, 1:])/2
        p_up = (p + director[1, :, :] * radius) / self.reference_length
        p_down = (p - director[1, :, :] * radius) / self.reference_length
        p_left = (p + director[0, :, :] * radius) / self.reference_length
        p_right = (p - director[0, :, :] * radius) / self.reference_length
        
        self.ax_rod.plot(p_center[0], p_center[1], p_center[2], color=self.rod_color if color is None else color, linestyle="--")
        return self.ax_rod

    def plot_strains(self, shear, kappa, color=None):
        _, voronoi_dilatation = calculate_dilatation(shear)
        curvature = kappa_to_curvature(kappa, voronoi_dilatation)

        for index_i in range(3):
            self.axes_shear[index_i].plot(
                self.s_shear,
                shear[index_i],
                color=self.rod_color if color is None else color
            )
            self.axes_curvature[index_i].plot(
                self.s_curvature,
                kappa[index_i],
                color=self.rod_color if color is None else color
            )
        return self.axes_shear, self.axes_curvature

    def plot_loads(self, external_force, external_couple):
        for index_i in range(3):
            self.axes_external_force[index_i].plot(
                self.s_external_force,
                np.zeros(self.s_external_force.shape),
                color="grey",
                linestyle="--"
            )
            self.axes_external_force[index_i].plot(
                self.s_external_force,
                external_force[index_i],
                color=self.rod_color
            )
            self.axes_external_couple[index_i].plot(
                self.s_external_couple,
                np.zeros(self.s_external_couple.shape),
                color="grey",
                linestyle="--"
            )
            self.axes_external_couple[index_i].plot(
                self.s_external_couple,
                external_couple[index_i],
                color=self.rod_color
            )
        return self.axes_external_force, self.axes_external_couple        

    def set_ax_rod_lim(self, x_lim=[-1.1, 1.1], y_lim=[-1.1, 1.1], z_lim=[-1.1, 1.1]):
        self.ax_rod.set_xlim(x_lim)
        self.ax_rod.set_ylim(y_lim)
        if self.ax3d_flag:
            self.ax_rod.set_zlim(z_lim)

    def set_ax_strains_lim(self, axes_shear_lim=None, axes_curvature_lim=None,):
        if axes_shear_lim is None:
            axes_shear_lim = [
                [-0.11, 0.11],
                [-0.11, 0.11],
                [-0.1, 2.1]
            ]
        if axes_curvature_lim is None:
            axes_curvature_lim = [
                [-110, 110],
                [-110, 110],
                [-11, 11],
            ]
        for index_i in range(3):
            shear_mean = np.average(axes_shear_lim[index_i])
            shear_log = np.floor(np.log10(axes_shear_lim[index_i][1] - shear_mean))
            curvature_mean = np.average(axes_curvature_lim[index_i])
            curvature_log = np.floor(np.log10(axes_curvature_lim[index_i][1] - curvature_mean))
            
            self.axes_shear[index_i].set_ylim(axes_shear_lim[index_i])
            self.axes_curvature[index_i].set_ylim(axes_curvature_lim[index_i])
            self.axes_shear[index_i].ticklabel_format(axis='y', scilimits=(shear_log, shear_log), useOffset=shear_mean)
            self.axes_curvature[index_i].ticklabel_format(axis='y', scilimits=(curvature_log, curvature_log), useOffset=curvature_mean)

    def set_ax_loads_lim(self, axes_external_force_lim=None, axes_external_couple_lim=None):
        if axes_external_force_lim is None:
            axes_external_force_lim = [
                [-0.11, 0.11],
                [-0.11, 0.11],
                [-0.11, 0.11]
            ]
        if axes_external_couple_lim is None:
            axes_external_couple_lim = [
                [-0.0011, 0.0011],
                [-0.0011, 0.0011],
                [-0.0011, 0.0011],
            ]
        for index_i in range(3):
            self.axes_external_force[index_i].set_ylim(axes_external_force_lim[index_i])
            self.axes_external_couple[index_i].set_ylim(axes_external_couple_lim[index_i])
            self.axes_external_force[index_i].ticklabel_format(axis='y', scilimits=(-1, -1))
            self.axes_external_couple[index_i].ticklabel_format(axis='y', scilimits=(-3, -3))

    def set_labels(self, time=None):
        if time is not None:
            self.ax_rod.set_title("time={:.2f} [sec]".format(time), fontsize=self.fontsize)
        for index_i in range(3):
            self.axes_curvature[index_i].set_ylabel(
                "    d$_{}$".format(index_i+1),
                fontsize=self.fontsize,
                rotation=0
            )
            self.axes_curvature[index_i].yaxis.set_label_position("right")

        self.axes_shear[0].set_title("shear", fontsize=self.fontsize)
        ylim = self.axes_shear[2].get_ylim()
        ylim_mean = np.average(ylim)
        position = 0.9 * (ylim[1] - ylim_mean) + ylim_mean
        self.axes_shear[2].text(0, position, 'stretch', fontsize=self.fontsize, ha='left', va='top')
        self.axes_shear[2].set_xlabel("$s$", fontsize=self.fontsize)
        
        self.axes_curvature[0].set_title("curvature", fontsize=self.fontsize)
        ylim = self.axes_curvature[2].get_ylim()
        ylim_mean = np.average(ylim)
        position = 0.9 * (ylim[1] - ylim_mean) + ylim_mean
        self.axes_curvature[2].text(1, position, 'twist', fontsize=self.fontsize, ha='right', va='top')
        self.axes_curvature[2].set_xlabel("$s$", fontsize=self.fontsize)
