#!/usr/bin/env python

import numpy as np


def propagation_matrix():
    pass


def delta_matrix():
    pass


def dynamic_matrix():
    pass


def dynamic_matrix_isotropic():
    pass


class Layer:
    '''Any coherent layer with a definite propagation matrix
    Matrices must be callables matrix(omega, k_trans)'''
    def __init__(self, propagation_matrix,
                 dynamic_matrix_l=None,
                 dynamic_matrix_r=None):
        self.propagation_matrix = propagation_matrix
        self.dynamic_matrix_l = dynamic_matrix_l \
            if dynamic_matrix_l is not None else \
            np.eye(4)
        self.dynamic_matrix_r = dynamic_matrix_r \
            if dynamic_matrix_r is not None else \
            np.eye(4)

    def jones_matrices(self, omega, k_trans):
        M = np.inv(self.dynamic_matrix_l(omega, k_trans)) @ \
            self.propagation_matrix(omega, k_trans) @ \
            self.dynamic_matrix_r(omega, k_trans)
        T_l = np.inv(M[2:4, 2:4])
        R_l = M[:2, 2:4] @ T_l
        R_r = -np.inv(M[2:4, 2:4]) @ M[2:4, :2]
        T_r = M[:2, :2] + M[:2, 2:4] @ R_r
        return T_l, R_l, T_r, R_r

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


class RepeatedLayer(Layer):
    def __init__(self, layer, number):
        self.layer = layer
        self.number = number

    def propagation_matrix(self):
        pass

    def dynamic_matrix_l(self, omega, k_trans):
        return self.layer.dynamic_matrix_l(omega, k_trans)

    def dynamic_matrix_r(self, omega, k_trans):
        return self.layer.dynamic_matrix_r(omega, k_trans)


class BerremanLayer(Layer):
    '''Layer with a Berreman delta matrix
    delta must be callable delta(omega)'''
    def __init__(self, delta_matrix, thickness):
        self.delta_matrix = delta_matrix
        self.thickness = thickness

    def propagation_matrix(self, omega, k_trans):
        return np.expm(1j *
                       omega *
                       self.thickness *
                       self.delta_matrix(omega))

    def dynamic_matrix(self, omega, k_trans):
        pass

    def dynamic_matrix_l(self, omega, k_trans):
        return self.dynamic_matrix_l

    def dynamic_matrix_r(self, omega, k_trans):
        return self.dynamic_matrix_r


class PermittivityLayer(BerremanLayer):
    '''Layer with a permittivity tensor
    permittivity must be callable perm(omega)'''
    def __init__(self, permittivity, thickness):
        self.permittivity = permittivity
        self.thickness = thickness

    def delta_matrix(self, omega, k_trans):
        ro = np.array([[0, 1], [-1, 0]])
        p_proj = np.outer(k_trans, k_trans)
        s_proj = - ro @ p_proj @ ro
        eo = self.permittivity_tensor[:2, :2]
        ev = self.permittivity_tensor[:2, 2:3]
        eh = self.permittivity_tensor[2:3, :2]
        ez = self.permittivity_tensor[2, 2]
        return np.block([[-np.outer(k_trans, eh)/ez,
                          np.eye(2) - p_proj/ez],
                         [eo - np.outer(ev, eh)/ez - s_proj,
                          - np.outer(ev, k_trans)/ez]])


class IsotropicLayer(PermittivityLayer):
    '''Isotropic layer defined by an index of refraction
    index must be callable index(omega)'''
    def __init__(self, index, thickness):
        self.index = index
        self.thickness = thickness

    def permittivity_tensor(self, omega):
        return self.index(omega)**2 * np.eye(3)

    def dynamic_matrix(self, omega, k_trans):
        N = k_trans
        n = self.index
        ssqr = np.dot(N, N)
        c = np.sqrt(1-ssqr/n**2)
        p_proj = np.array([[1, 0], [0, 0]]) \
            if not ssqr else np.outer(N, N) / ssqr
        s_proj = np.eye(2) - p_proj
        return np.block([[c*p_proj + s_proj, c*p_proj + s_proj],
                        [n*(p_proj + c*s_proj), -n*(p_proj + c*s_proj)]])

    def rotate_2d(self, angle, inplace=False):
        if not inplace:
            return self
