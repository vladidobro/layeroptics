import equinox as eqx
import jax
import jax.numpy as jnp
import functools as ft
from typing import Protocol
from jaxtyping import Float, Array, Integer

Wavelength = Float[Array, ""]
Wavevector = Float[Array, "2"]
PropagationMatrix = Float[Array, "4 4"]
Grassmanian = Float[Array, "4 4"]


class Environment(Protocol):
    def left(self, k0: Wavelength, k: Wavevector) -> Grassmanian:
        ...

    def right(self, k0: Wavelength, k: Wavevector) -> Grassmanian:
        ...

class Layer(Protocol):
    def __call__(self, k0: Wavelength, k: Wavevector) -> PropagationMatrix:
        ...

class MultiLayer(eqx.Module):
    layers: list[Layer]

    def __call__(self, k0: Wavelength, k: Wavevector) -> PropagationMatrix:
        return ft.reduce(jnp.matmul, (layer(k0, k) for layer in self.layers))

class RepeatedLayer(eqx.Module):
    layer: Layer
    repetitions: Integer

    def __call__(self, k0: Wavelength, k: Wavevector) -> PropagationMatrix:
        return jnp.linalg.matrix_power(self.layer(k0, k), self.repetitions)

class IsotropicLayer(eqx.Module):
    eps: Float[Array, ""]

    def __call__(self, k0: Wavelength, k: Wavevector) -> PropagationMatrix:
        return jnp.array(0, 0, )
        return self.n * jnp.identity(4)

def reflect(struct: Layer, env_l: Environment, env_r: Environment) -> tuple[PropagationMatrix, PropagationMatrix]:
    ...

struct = RepeatedLayer(
    layer=MultiLayer(layers=[
        IsotropicLayer(n=jnp.array(1)),
        IsotropicLayer(n=jnp.array(2)),
    ]),
    repetitions=30,
)

struct(jnp.array([1]), jnp.array([2, 3]))
