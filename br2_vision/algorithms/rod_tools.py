"""
Created on Oct. 19, 2020
@author: Heng-Sheng (Hanson) Chang
"""

import os
import sys

import numpy as np
from elastica._calculus import difference_kernel, quadrature_kernel
from elastica._linalg import (
    _batch_cross,
    _batch_matrix_transpose,
    _batch_matvec,
    _batch_norm,
)
from elastica.external_forces import NoForces
from numba import jit, njit


@njit(cache=True)
def inverse_rigidity_matrix(matrix):
    inverse_matrix = np.zeros(matrix.shape)
    for i in range(inverse_matrix.shape[2]):
        inverse_matrix[:, :, i] = np.linalg.inv(matrix[:, :, i])
    return inverse_matrix


@njit(cache=True)
def strain_to_shear_and_curvature(strain):
    n_elems = int((strain.size + 3) / 6)
    shear = np.zeros((3, n_elems))
    curvature = np.zeros((3, n_elems - 1))
    index = 0
    for i in range(3):
        shear[i, :] = strain[index : index + n_elems]
        index += n_elems
    for i in range(3):
        curvature[i] = strain[index : index + n_elems - 1]
        index += n_elems - 1
    return shear, curvature


@njit(cache=True)
def shear_and_curvature_to_strain(shear, curvature):
    strain = np.zeros(shear.size + curvature.size)
    n_elems = shear.shape[1]
    index = 0
    for i in range(3):
        strain[index : index + n_elems] = shear[i, :]
        index += n_elems
    for i in range(3):
        strain[index : index + n_elems - 1] = curvature[i]
        index += n_elems - 1
    return strain


@njit(cache=True)
def _lab_to_material(directors, lab_vectors):
    return _batch_matvec(directors, lab_vectors)


@njit(cache=True)
def _material_to_lab(directors, material_vectors):
    blocksize = material_vectors.shape[1]
    output_vector = np.zeros((3, blocksize))
    for i in range(3):
        for j in range(3):
            for k in range(blocksize):
                output_vector[i, k] += directors[j, i, k] * material_vectors[j, k]
    return output_vector


@njit(cache=True)
def average2D(vector_collection):
    blocksize = vector_collection.shape[1] - 1
    output_vector = np.zeros((3, blocksize))
    for k in range(blocksize):
        for i in range(3):
            output_vector[i, k] = (
                vector_collection[i, k] + vector_collection[i, k + 1]
            ) / 2
    return output_vector


@njit(cache=True)
def average1D(vector_collection):
    blocksize = vector_collection.shape[0] - 1
    output_vector = np.zeros(blocksize)
    for k in range(blocksize):
        output_vector[k] = (vector_collection[k] + vector_collection[k + 1]) / 2
    return output_vector


@njit(cache=True)
def distance(position_collection, target_position):
    blocksize = position_collection.shape[1]
    distance_collection = np.zeros(blocksize)
    for k in range(blocksize):
        distance_collection[k] = (
            (position_collection[0, k] - target_position[0]) ** 2
            + (position_collection[1, k] - target_position[1]) ** 2
            + (position_collection[2, k] - target_position[2]) ** 2
        ) ** 0.5
    return distance_collection


@njit(cache=True)
def forward_path(dl, shear, kappa, position_collection, director_collection):
    # _, voronoi_dilatation = calculate_dilatation(shear)
    # curvature = kappa_to_curvature(kappa, voronoi_dilatation)
    for i in range(dl.shape[0] - 1):
        next_position(
            director_collection[:, :, i],
            shear[:, i] * dl[i],
            position_collection[:, i : i + 2],
        )
        next_director(kappa[:, i] * dl[i], director_collection[:, :, i : i + 2])
    next_position(
        director_collection[:, :, -1],
        shear[:, -1] * dl[-1],
        position_collection[:, -2:],
    )


@njit(cache=True)
def next_position(director, delta, positions):
    positions[:, 1] = positions[:, 0]
    for index_i in range(3):
        for index_j in range(3):
            positions[index_i, 1] += director[index_j, index_i] * delta[index_j]
    return


@njit(cache=True)
def next_director(rotation, directors):
    Rotation = get_rotation_matrix(rotation)
    for index_i in range(3):
        for index_j in range(3):
            directors[index_i, index_j, 1] = 0
            for index_k in range(3):
                directors[index_i, index_j, 1] += (
                    Rotation[index_k, index_i] * directors[index_k, index_j, 0]
                )
    return


@njit(cache=True)
def get_rotation_matrix(axis):
    angle = np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)

    axis = axis / (angle + 1e-8)
    K = np.zeros((3, 3))
    K[2, 1] = axis[0]
    K[1, 2] = -axis[0]
    K[0, 2] = axis[1]
    K[2, 0] = -axis[1]
    K[1, 0] = axis[2]
    K[0, 1] = -axis[2]

    K2 = np.zeros((3, 3))
    K2[0, 0] = -(axis[1] * axis[1] + axis[2] * axis[2])
    K2[1, 1] = -(axis[2] * axis[2] + axis[0] * axis[0])
    K2[2, 2] = -(axis[0] * axis[0] + axis[1] * axis[1])
    K2[0, 1] = axis[0] * axis[1]
    K2[1, 0] = axis[0] * axis[1]
    K2[0, 2] = axis[0] * axis[2]
    K2[2, 0] = axis[0] * axis[2]
    K2[1, 2] = axis[1] * axis[2]
    K2[2, 1] = axis[1] * axis[2]

    Rotation = np.sin(angle) * K + (1 - np.cos(angle)) * K2
    Rotation[0, 0] += 1
    Rotation[1, 1] += 1
    Rotation[2, 2] += 1

    return Rotation


@njit(cache=True)
def backward_path(dl, director, shear, ns, ms, n1, m1, internal_force, internal_couple):
    n = np.zeros(internal_force.shape)
    n[:, -1] = n1
    for i in range(n.shape[1] - 1):
        n[:, -1 - i - 1] = n[:, -1 - i] - (ns[:, -1 - i] + ns[:, -1 - i - 1]) * dl / 2
    internal_force[:, :] = _batch_matvec(director, n)

    m = np.zeros(internal_force.shape)
    m[:, -1] = m1
    ms[:, :] = -_batch_cross(_batch_matvec(_batch_matrix_transpose(director), shear), n)
    for i in range(m.shape[1] - 1):
        m[:, -1 - i - 1] = m[:, -1 - i] - (ms[:, -1 - i] + ms[:, -1 - i - 1]) * dl / 2
    internal_couple[:, :] = quadrature_kernel(_batch_matvec(director, m))[:, 1:-1]
    return


@njit(cache=True)
def calculate_dilatation(shear):
    dilatation = _batch_norm(shear)
    voronoi_dilatation = (dilatation[:-1] + dilatation[1:]) / 2
    return dilatation, voronoi_dilatation


@njit(cache=True)
def calculate_length(position):
    length = 0
    for i in range(position.shape[1] - 1):
        length += np.sqrt(
            (position[0, i + 1] - position[0, i]) ** 2
            + (position[1, i + 1] - position[1, i]) ** 2
            + (position[2, i + 1] - position[2, i]) ** 2
        )
    return length


@njit(cache=True)
def sigma_to_shear(sigma):
    shear = np.zeros(sigma.shape)
    for i in range(shear.shape[1]):
        shear[0, i] = sigma[0, i]
        shear[1, i] = sigma[1, i]
        shear[2, i] = sigma[2, i] + 1
    return shear


@njit(cache=True)
def kappa_to_curvature(kappa, voronoi_dilatation):
    curvature = np.zeros(kappa.shape)
    for i in range(curvature.shape[1]):
        curvature[0, i] = kappa[0, i] / voronoi_dilatation[i]
        curvature[1, i] = kappa[1, i] / voronoi_dilatation[i]
        curvature[2, i] = kappa[2, i] / voronoi_dilatation[i]
    return curvature
