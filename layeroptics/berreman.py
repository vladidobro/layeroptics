#!/usr/bin/env python

import numpy as np
from .common import *


class Layer:
    '''Any coherent layer with a definite propagation matrix
    Matrices must be callables matrix(omega, k_trans)'''
    def __init__(self, propagation_matrix,
                 dynamic_matrix_l=None,
                 dynamic_matrix_r=None):
        self.propagation_matrix = propagation_matrix
        self.dynamic_matrix_l = dynamic_matrix_l
        self.dynamic_matrix_r = dynamic_matrix_r
        if dynamic_matrix_l is None:
            self.dynamic_matrix_l = lambda omega, k_trans: \
                dynamic_matrix_from_propagation(
                        self.propagation_matrix(omega, k_trans),
                        omega,
                        k_trans)
        if dynamic_matrix_r is None:
            self.dynamic_matrix_r = lambda omega, k_trans: \
                dynamic_matrix_from_propagation(
                        self.propagation_matrix(omega, k_trans),
                        omega,
                        k_trans)

    def jones_matrices(self, omega, k_trans):
        return jones_matrices_from_modal(modal_matrix(
            self.propagation_matrix(omega, k_trans),
            self.dynamic_matrix_l(omega, k_trans),
            self.dynamic_matrix_r(omega, k_trans)))

    def rotate_2d(self, angle, inplace=False):
        pass


class MultiLayer(Layer):
    def __init__(self, layers):
        self.layers = layers

    def propagation_matrix(self, omega, k_trans):
        pass

    def dynamic_matrix_l(self, omega, k_trans):
        return self.layers[0].dynamic_matrix_l(omega, k_trans)

    def dynamic_matrix_r(self, omega, k_trans):
        return self.layers[-1].dynamic_matrix_r(omega, k_trans)

    def rotate_2d(self, angle, inplace=False):
        pass


class RepeatedLayer(Layer):
    def __init__(self, layer, number):
        self.layer = layer
        self.number = number

    def propagation_matrix(self, omega, k_trans):
        pass

    def dynamic_matrix_l(self, omega, k_trans):
        return self.layer.dynamic_matrix_l(omega, k_trans)

    def dynamic_matrix_r(self, omega, k_trans):
        return self.layer.dynamic_matrix_r(omega, k_trans)

    def rotate_2d(self, angle, inplace=False):
        pass


class BerremanLayer(Layer):
    '''Layer with a Berreman delta matrix
    delta must be callable delta(omega, k_trans)'''
    def __init__(self, delta_matrix, thickness):
        self.delta_matrix = delta_matrix
        self.thickness = thickness

    def propagation_matrix(self, omega, k_trans):
        return propagation_matrix(self.delta_matrix(omega, k_trans),
                                  omega, self.thickness)

    def dynamic_matrix(self, omega, k_trans):
        return dynamic_matrix_from_delta(self.delta_matrix(omega, k_trans),
                                         omega, k_trans)

    def dynamic_matrix_l(self, omega, k_trans):
        return self.dynamic_matrix(omega, k_trans)

    def dynamic_matrix_r(self, omega, k_trans):
        return self.dynamic_matrix(omega, k_trans)

    def rotate_2d(self, angle, inplace=False):
        pass


class PermittivityLayer(BerremanLayer):
    '''Layer with a permittivity tensor
    permittivity must be callable perm(omega)'''
    def __init__(self, permittivity, thickness):
        self.permittivity = permittivity
        self.thickness = thickness

    def delta_matrix(self, omega, k_trans):
        return delta_matrix(self.permittivity,
                            omega, k_trans)

    def rotate_2d(self, angle, inplace=False):
        pass

    def rotate_3d(self, rotation_matrix, inplace=False):
        pass


class IsotropicLayer(PermittivityLayer):
    '''Isotropic layer defined by an index of refraction
    index must be callable index(omega)'''
    def __init__(self, index, thickness):
        self.index = index
        self.thickness = thickness

    def permittivity_tensor(self, omega):
        return self.index(omega)**2 * np.eye(3)

    def dynamic_matrix(self, omega, k_trans):
        return delta_matrix_isotropic(self.index(omega),
                                      omega, k_trans)

    def rotate_2d(self, angle, inplace=False):
        if not inplace:
            return self

    def rotate_3d(self, rotation_matrix, inplace=False):
        if not inplace:
            return self
