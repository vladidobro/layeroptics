import pandas as pd
import numpy as np
from functools import reduce

from layeroptics.layer import MultiLayer, PermittivityLayer, RepeatedLayer, IsotropicLayer
from layeroptics.compute import jones_from_field, jones_to_angles, layer_TR_field_l, mo_permittivity, mo_linear_tensor, mo_quadratic_tensor, subsp_from_base, vector_3d_from_angles

def mo_layer(m_angle):
    magnetization = vector_3d_from_angles(m_angle, 0)

    mo_perm = mo_permittivity(
        [
            (0, None),
            (1, mo_linear_tensor(None)),
            (2, mo_quadratic_tensor(None)),
        ],
        magnetization
    )

    return MultiLayer([
        PermittivityLayer(
            mo_perm,
            5
        ),
        RepeatedLayer(MultiLayer([
            IsotropicLayer(1.5, 1),
            IsotropicLayer(1.7, 1),
        ]), 50)
    ])

def experiment(light_angle, m_angle):
    layer = mo_layer(m_angle)
    k = NotImplemented
    light_in = NotImplemented
    vacuum_base = NotImplemented

    subsp = subsp_from_base(vacuum_base)
    _, R = layer_TR_field_l(k, layer, subsp, subsp)
    J_R = jones_from_field(R, vacuum_base, vacuum_base)
    light_out = J_R * light_in
    light_angle_out = jones_to_angles(light_out).polarization

    return light_angle_out

def cross(*series):
    return reduce(lambda s1, s2: s1.to_frame().merge(s2.to_frame(), how='cross'), series)

def apply(df, f):
    return df.apply(lambda r: f(**r), axis=1)

df = cross(
    pd.Series(np.linspace(0, 180, 180, endpoint=False), name='light_angle'),
    pd.Series(np.linspace(0, 360, 360, endpoint=False), name='m_angle'),
)

df['light_angle_out'] = apply(df, experiment)
