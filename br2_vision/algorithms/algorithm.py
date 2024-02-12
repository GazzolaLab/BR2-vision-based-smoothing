"""
Created on Oct. 19, 2020
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np

import os, sys

from rod_tools import sigma_to_shear


class Algorithm(object):
    def __init__(self, rod, algo_config):

        self.rod = rod
        self.config = algo_config

        """
        make a copy of the required info of the rod 
        so that it will not affect the values of the original copy
        """

        self.n_elems = rod.n_elems
        self.reference_length = np.sum(rod.rest_lengths)

        self.dl = rod.rest_lengths.copy()
        self.ds = self.dl / self.reference_length
        self.s = np.insert(np.cumsum(self.ds), 0, 0)
        self.s_position = self.s.copy()
        self.s_director = (self.s[:-1] + self.s[1:]) / 2
        self.s_shear = (self.s[:-1] + self.s[1:]) / 2
        self.s_kappa = self.s[1:-1]

        self.position = rod.position_collection.copy()
        self.director = rod.director_collection.copy()

        self.shear = sigma_to_shear(rod.sigma)
        self.rest_shear = sigma_to_shear(rod.rest_sigma)
        self.sigma = rod.sigma.copy()
        self.rest_sigma = rod.rest_sigma.copy()
        self.kappa = rod.kappa.copy()
        self.rest_kappa = rod.rest_kappa.copy()

        self.radius = rod.radius.copy()
        self.shear_matrix = rod.shear_matrix.copy()
        self.bend_matrix = rod.bend_matrix.copy()

        self.internal_force = np.zeros((3, self.n_elems))
        self.internal_couple = np.zeros((3, self.n_elems - 1))

    def run(self, plot_flag=False):
        raise NotImplementedError
