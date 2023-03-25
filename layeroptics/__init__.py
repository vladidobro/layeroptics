#!/usr/bin/env python

import numpy as np
import matplotlib as plt
from time import sleep

from numpy import array, eye, outer, zeros, pi
from numpy.linalg import inv, matrix_power
from scipy.linalg import expm
from functools import reduce

class JonesVector:
    def __init__(self,jones_vector):
        self.jones_vector = array(jones_vector)
    
    PAULI_MATRICES = array([eye(2),[[1,0],[0,-1]],[[0,1],[1,0]],[[0,-1j],[1j,0]]])
    
    @property
    def stokes_vector(self):
        return array(list(map(lambda sigma: np.dot(self.jones_vector.conj(), sigma @ self.jones_vector), 
                              self.PAULI_MATRICES)))
    
    @property
    def rotation(self):
        return np.angle(self.stokes_vector[1] + 1j*self.stokes_vector[2]) / 2
    
    @property
    def ellipticity(self):
        return np.angle(abs(self.stokes_vector[1] + 1j*self.stokes_vector[2])
                        + 1j*self.stokes_vector[3]) / 2
    
    @property
    def intensity(self):
        return stokes_vector[0]
    
    @property
    def phase(self):
        # not mmplemented
        pass

class CoherentStructure:
    vacuum_wavevector = Parameter(Listener('k0'))
    trans_wavevector_normalized = Parameter(Listener('N'))
    propagation_matrix = Parameter(eye(4))
    dynamic_in = Parameter(eye(4))
    dynamic_out = Parameter(eye(4))
    
    def __init__(self, topic):
        self.topic = topic
    
    @property
    def jones_matrices(self):
        M = inv(self.dynamic_in) @ self.propagation_matrix @ self.dynamic_out
        T_in = inv(M[2:4,2:4])
        R_in = M[:2,2:4] @ T_in
        R_out = -inv(M[2:4,2:4]) @ M[2:4,:2]
        T_out = M[:2,:2] + M[:2,2:4] @ R_out
        return T_in, R_in, T_out, R_out
    
class MixedThickStructure:
    # not implemented
    def __init__(self,layers):
        pass

class MultiLayer(CoherentStructure):
    
    layers = Parameter([])
    
    def __init__(self, topic, layers=None):
        self.topic = topic
        self.layers = [] if layers is None else layers
        
    @property
    def propagation_matrix(self):
        return reduce(lambda x,y: x @ y, map(lambda x: x.propagation_matrix, self.layers), eye(4))

    @property
    def dynamic_in(self):
        return eye(4) if not self.layers else self.layers[0].dynamic_in

    @property
    def dynamic_out(self):
        return eye(4) if not self.layers else self.layers[-1].dynamic_out

class RepeatedLayer(CoherentStructure):
    
    layer = Parameter()
    number = Parameter(0)
    
    def __init__(self, topic, layer, number=0):
        self.topic = topic
        self.layer = layer
        self.number = number
    
    @property
    def propagation_matrix(self):
        return matrix_power(self.layer.propagation_matrix, self.number)
    
    @property
    def dynamic_in(self):
        return self.layer.dynamic_in
    
    @property
    def dynamic_out(self):
        return self.layer.dynamic_out

class BerremanLayer(CoherentStructure):
    
    berreman_matrix = Parameter(eye(4))
    thickness = Parameter(0)
    
    def __init__(self, topic, berreman_matrix, thickness=0):
        self.topic = topic
        self.berreman_matrix = berreman_matrix
        self.thickness = thickness
    
    @property
    def propagation_matrix(self):
        return expm(1j * self.vacuum_wavevector * self.thickness * self.berreman_matrix)
    
    @property
    def dynamic_matrix(self):
        # not implemented
        return eye(4)
    
    @property
    def dynamic_in(self):
        return self.dynamic_matrix
    
    @property
    def dynamic_out(self):
        return self.dynamic_matrix
    

class PermittivityLayer(BerremanLayer):
    
    permittivity_tensor = Parameter(3*eye(3))
    thickness = Parameter(0)
    
    def __init__(self, topic, permittivity_tensor, thickness=0):
        self.topic = topic
        self.permittivity_tensor = permittivity_tensor
        self.thickness = thickness
    
    @property
    def berreman_matrix(self):
        N = self.trans_wavevector_normalized
        ro = array([[0,1],[-1,0]])
        p_proj = outer(N,N)
        s_proj = - ro @ p_proj @ ro
        eo = self.permittivity_tensor[:2,:2]
        ev = self.permittivity_tensor[:2,2:3]
        eh = self.permittivity_tensor[2:3,:2]
        ez = self.permittivity_tensor[2,2]
        return np.block([[ -outer(N,eh)/ez, eye(2) - p_proj/ez],
                         [ eo - outer(ev,eh)/ez - s_proj, -outer(ev,N)/ez]])

class IsotropicLayer(PermittivityLayer):
    
    index_of_refraction = Parameter(1)
    thickness = Parameter(0)
    
    def __init__(self, topic, index_of_refraction=1, thickness=0):
        self.topic = topic
        self.index_of_refraction = index_of_refraction
        self.thickness = thickness
    
    @property
    def permittivity_tensor(self):
        return self.index_of_refraction**2 * eye(3)
    
    @property
    def dynamic_matrix(self):
        N = self.trans_wavevector_normalized
        if not N.imag.any():
            n = self.index_of_refraction
            ssqr = np.dot(N,N)
            c = np.sqrt(1-ssqr/n**2)
            p_proj = array([[1,0],[0,0]]) if not ssqr else outer(N,N)/ssqr
            s_proj = eye(2) - p_proj
            return np.block([[ c*p_proj + s_proj, c*p_proj + s_proj],
                         [ n*(p_proj + c*s_proj), -n*(p_proj + c*s_proj)]])
        else:
            return super().dynamic_matrix

class MagnetoopticLayer(PermittivityLayer):
    # not implemented
    pass
