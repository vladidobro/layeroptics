#!/usr/bin/env python

import numpy as np

pauli_matrices = np.array([np.eye(2),
                           [[1, 0], [0, -1]],
                           [[0, 1], [1, 0]],
                           [[0, -1j], [1j, 0]]])


def stokes_from_jones(j_vec):
    return np.array(list(map(
            lambda sigma: np.dot(j_vec.conj(),
                                 sigma @ j_vec),
            pauli_matrices)))


def jones_from_stokes(s_vec, phase):
    pass


def phase_from_jones(j_vec):
    return 0


def angle_from_stokes(s_vec):
    return np.angle(s_vec[1] + 1j*s_vec[2]) / 2


def ellipticity_from_stokes(s_vec):
    return np.angle(abs(s_vec[1] + 1j*s_vec[2])
                    + 1j*s_vec[3]) / 2


def intensity_from_stokes(s_vec):
    return s_vec[0]


def rotation_matrix_2d(angle):
    '''Angles always in radian'''
    return np.array([0])


def rotation_matrix_3d(precession_angle, nutation_angle, internal_angle):
    '''Angles in radian'''
    return np.eye(3)


def propagation_matrix(delta_matrix, omega, thickness):
    return np.expm(1j *
                   omega *
                   thickness *
                   delta_matrix)


def delta_matrix(permittivity, omega, k_trans):
    ro = np.array([[0, 1], [-1, 0]])
    p_proj = np.outer(k_trans, k_trans)
    s_proj = - ro @ p_proj @ ro
    eo = permittivity[:2, :2]
    ev = permittivity[:2, 2:3]
    eh = permittivity[2:3, :2]
    ez = permittivity[2, 2]
    return np.block([[-np.outer(k_trans, eh)/ez,
                      np.eye(2) - p_proj/ez],
                     [eo - np.outer(ev, eh)/ez - s_proj,
                      - np.outer(ev, k_trans)/ez]])


def dynamic_matrix_from_delta(delta_matrix, omega, k_trans):
    pass


def dynamic_matrix_from_propagation(propagation_matrix, omega, k_trans):
    pass


def dynamic_matrix_isotropic(index, omega, k_trans):
    ssqr = np.dot(k_trans, k_trans)
    c = np.sqrt(1-ssqr/index**2)
    p_proj = np.array([[1, 0], [0, 0]]) \
        if not ssqr else np.outer(k_trans, k_trans) / ssqr
    s_proj = np.eye(2) - p_proj
    return np.block([[c*p_proj + s_proj, c*p_proj + s_proj],
                    [index*(p_proj + c*s_proj), -index*(p_proj + c*s_proj)]])


def modal_matrix(propagation, dynamic_l, dynamic_r):
    return np.inv(dynamic_l) @ propagation @ dynamic_r


def jones_matrices_from_modal(modal_matrix):
    T_l = np.inv(modal_matrix[2:4, 2:4])
    R_l = modal_matrix[:2, 2:4] @ T_l
    R_r = -np.inv(modal_matrix[2:4, 2:4]) @ modal_matrix[2:4, :2]
    T_r = modal_matrix[:2, :2] + modal_matrix[:2, 2:4] @ R_r
    return T_l, R_l, T_r, R_r
