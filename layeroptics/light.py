#!/usr/bin/env python

import numpy as np

class JonesVector:
    def __init__(self, j_vec):
        self.j_vec = np.array(j_vec)

    def to_stokes(self, only_vec=False):
        s_vec = np.array(list(map(
            lambda sigma: np.dot(self.j_vec.conj(),
                                 sigma @ self.j_vec),
            PAULI_MATRICES)))
        if only_vec:
            return s_vec
        else:
            return StokesVector(s_vec)

    @property
    def angle(self):
        s_vec = self.to_stokes(only_vec=True)
        return np.angle(s_vec[1]+1j*s_vec[2]) / 2

    @property
    def ellipticity(self):
        s_vec = self.to_stokes(only_vec=True)
        return np.angle(abs(s_vec[1]+1j*s_vec[2])
                        + 1j*s_vec[3]) / 2

    @property
    def intensity(self):
        s_vec = self.to_stokes(only_vec=True)
        return s_vec[0]

    @property
    def phase(self):
        # not mmplemented
        return 0


class StokesVector:
    def __init__(self, s_vec):
        self.s_vec = np.array(s_vec)

    def to_jones(self, only_vec=False):
        j_vec = np.array([
                0,
                0,
                0,
                0])
        if only_vec:
            return j_vec
        else:
            return JonesVector(j_vec)

    @property
    def rotation(self):
        return np.angle(self.s_vec[1] + 1j*self.s_vec[2]) / 2

    @property
    def ellipticity(self):
        return np.angle(abs(self.s_vec[1] + 1j*self.s_vec[2])
                        + 1j*self.s_vec[3]) / 2

    @property
    def intensity(self):
        return self.s_vec[0]
