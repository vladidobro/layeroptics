from dataclasses import dataclass
from typing import List, Tuple

from . import compute
from .types import Transfer, Delta, Permittivity, Wavevector, Subspace, Magnetization, MOTensor


from typing import Protocol

class HasTransferMatrix(Protocol):
    def transfer(self, k: Wavevector) -> Transfer:
        ...

@dataclass 
class RotatedLayer:
    layer: HasTransferMatrix
    angle: float

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.rotate_transfer(
                self.layer.transfer(k),
                self.angle
        )

@dataclass
class MultiLayer:
    layers: List[HasTransferMatrix]

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_multi(k, (l.transfer(k) for l in self.layers))

@dataclass
class RepeatedLayer:
    layer: HasTransferMatrix
    n: int

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_repeat(k, self.layer.transfer(k), self.n)

@dataclass
class BerremanLayer:
    delta: Delta
    thickness: float

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_berreman(k, self.delta, self.thickness)

@dataclass
class IsotropicLayer:
    index: float
    thickness: float

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_berreman(
                k, 
                compute.delta_from_isotropic(self.index), 
                self.thickness,
        )

@dataclass
class PermittivityLayer:
    permittivity: Permittivity
    thickness: float

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_berreman(
                k,
                compute.delta_from_permittivity(self.permittivity),
                self.thickness,
        )

@dataclass
class ActiveLayer:
    index: float
    activity: float
    thickness: float

    def transfer(self, k: Wavevector) -> Transfer:
        return compute.transfer_from_berreman(
                k,
                compute.delta_from_active(self.index, self.activity),
                self.thickness
        )

@dataclass
class MagnetoOpticLayer:
    permittivity: Permittivity
    magnetization: Magnetization
    mo_tensors: List[Tuple[int, MOTensor]]
    thickness: float

    def transfer(self, k: Wavevector) -> Transfer:
        ...
