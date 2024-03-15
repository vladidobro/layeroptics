from typing import Any, List, Tuple, Literal
from .types import TransmissionReflectionMatrices


def transfer_from_multi(k, props):
    return NotImplemented

def transfer_from_repeat(k, prop, n):
    return NotImplemented

def transfer_from_berreman(k, delta, thickness):
    return NotImplemented

def delta_from_permittivity(permittivity):
    return NotImplemented

def delta_from_isotropic(index):
    return NotImplemented

def delta_from_active(index, activity):
    return NotImplemented

def rotate_permittivity(rot, perm):
    return NotImplemented

def rotation_2d(angle):
    return NotImplemented

def rotate_transfer(transfer, angle):
    return NotImplemented

def mo_order_permittivity(order, mo_tensor, magnetization):
    return NotImplemented

def mo_permittivity(mo_tensors: List[Tuple[int, Any]], magnetization):
    return sum(mo_order_permittivity(order, mo_tensor, magnetization) for order, mo_tensor in mo_tensors)

def mo_linear_tensor(k_tensor):
    return NotImplemented

def mo_quadratic_tensor(g_tensor):
    return NotImplemented

def vector_3d_from_angles(precesion, nutation):
    return NotImplemented

def transmission_and_reflection_field_l(transfer, lsubsp, rsubsp):
    return NotImplemented

def transmission_and_reflection_field_r(transfer, lsubsp, rsubsp):
    return NotImplemented

def jones_from_field(transfer, inbase, outbase):
    return NotImplemented

def subsp_from_base(base):
    return NotImplemented

def jones_to_angles(jones):
    return NotImplemented

def layer_TR_field_l(k, layer, lsubsp, rsubsp):
    return TransmissionReflectionMatrices(*transmission_and_reflection_field_l(layer.transfer(k), lsubsp, rsubsp))

def layer_TR_field_r(k, layer, lsubsp, rsubsp):
    return TransmissionReflectionMatrices(*transmission_and_reflection_field_r(layer.transfer(k), lsubsp, rsubsp))
