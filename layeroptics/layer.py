from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, List, Literal, Tuple

from . import compute
from .types import Transfer, Delta, Permittivity, Wavevector, TransmissionReflectionMatrices, Subsp

class Layer(ABC):
    @abstractmethod
    def transfer(self, k: Wavevector) -> Transfer:
        return NotImplemented

    def rotate(self, angle: float) -> "RotatedLayer":
        return RotatedLayer(layer=self, angle=angle)

@dataclass 
class RotatedLayer(Layer):
    layer: Layer
    angle: float

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.rotate_transfer(
                self.layer.transfer(k),
                self.angle
        )

    def rotate(self, angle: float) -> "RotatedLayer":
        return RotatedLayer(layer=self.layer, angle=self.angle + angle)

@dataclass
class MultiLayer(Layer):
    layers: List[Layer]

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_multi(k, (l.transfer(k) for l in self.layers))

@dataclass
class RepeatedLayer(Layer):
    layer: Layer
    n: int

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_repeat(k, self.layer.transfer(k), self.n)

@dataclass
class BerremanLayer(Layer):
    delta: Delta
    thickness: float

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_berreman(k, self.delta, self.thickness)

@dataclass
class IsotropicLayer(Layer):
    index: float
    thickness: float

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_berreman(
                k, 
                compute.delta_from_isotropic(self.index), 
                self.thickness,
        )

@dataclass
class PermittivityLayer(Layer):
    permittivity: Permittivity
    thickness: float

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_berreman(
                k,
                compute.delta_from_permittivity(self.permittivity),
                self.thickness,
        )

@dataclass
class ActiveLayer(Layer):
    index: float
    activity: float
    thickness: float

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_berreman(
                k,
                compute.delta_from_active(self.index, self.activity),
                self.thickness
        )
