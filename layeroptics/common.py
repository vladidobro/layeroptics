#!/usr/bin/env python

import numpy as np

pauli_matrices = np.array([np.eye(2),
                           [[1, 0], [0, -1]],
                           [[0, 1], [1, 0]],
                           [[0, -1j], [1j, 0]]])


def rotation_matrix_2d(angle):
    '''Angles always in radian'''
    return np.array([0])


def propagation_matrix():
    pass


def delta_matrix():
    pass


def dynamic_matrix():
    pass


def dynamic_matrix_isotropic():
    pass
