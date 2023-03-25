#!/usr/bin/env python

import numpy as np
from .common import *


class JonesVector:
    def __init__(self, j_vec):
        self.j_vec = np.array(j_vec)

    def to_stokes(self, only_vec=False):
        s_vec = stokes_from_jones(self.j_vec)
        if only_vec:
            return s_vec
        else:
            return StokesVector(s_vec)

    @property
    def angle(self):
        s_vec = self.to_stokes(only_vec=True)
        return angle_from_stokes(s_vec)

    @property
    def ellipticity(self):
        s_vec = self.to_stokes(only_vec=True)
        return ellipticity_from_stokes(s_vec)

    @property
    def intensity(self):
        s_vec = self.to_stokes(only_vec=True)
        return intensity_from_stokes(s_vec)

    @property
    def phase(self):
        return phase_from_jones(self.j_vec)


class StokesVector:
    def __init__(self, s_vec):
        self.s_vec = np.array(s_vec)

    def to_jones(self, phase=0, only_vec=False):
        j_vec = jones_from_stokes(self.s_vec, phase)
        if only_vec:
            return j_vec
        else:
            return JonesVector(j_vec)

    @property
    def angle(self):
        return angle_from_stokes(self.s_vec)

    @property
    def ellipticity(self):
        return ellipticity_from_stokes(self.s_vec)

    @property
    def intensity(self):
        return intensity_from_stokes(self.s_vec)
