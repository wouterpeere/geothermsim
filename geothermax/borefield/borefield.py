# -*- coding: utf-8 -*-
from typing import List, Self

import jax
from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ..basis import Basis
from ..borehole import Borehole
from ..path import Path


class Borefield:

    def __init__(self, boreholes: List[Borehole]):
        self.boreholes = boreholes
        self.n_boreholes = len(boreholes)
        self.n_nodes = boreholes[0].n_nodes
        self.L = jnp.array([borehole.L for borehole in self.boreholes])

        # --- Basis functions ---
        self.f_psi = jit(
            lambda _eta: jnp.stack(
                [
                    vmap(borehole.f_psi, in_axes=0)(_eta) if len(jnp.shape(_eta)) > 0 else borehole.f_psi(_eta)
                    for borehole in boreholes
                ],
                axis=-2
            )
        )
        self.f = jit(
            lambda _eta, f_nodes: vmap(jnp.dot, in_axes=(-2, 0), out_axes=0)(self.f_psi(_eta), f_nodes)
        )

        # --- Nodal values of path and basis functions ---
        # Borehole coordinates (xi)
        self.xi = self.boreholes[0].xi
        # Positions (p)
        self.p = jnp.stack([borehole.p for borehole in self.boreholes], axis=0)
        # Derivatives of position (dp/dxi)
        self.dp_dxi = jnp.stack([borehole.dp_dxi for borehole in self.boreholes], axis=0)
        # Norms of the Jacobian (J)
        self.J = jnp.stack([borehole.J for borehole in self.boreholes], axis=0)
        # Longitudinal positions (s)
        self.s = jnp.stack([borehole.s for borehole in self.boreholes], axis=0)
        # Integration weights
        self.w = jnp.stack([borehole.w for borehole in self.boreholes], axis=0)

    def h_to_self(self, time: Array, alpha: float) -> Array:
        n_boreholes = self.n_boreholes
        h_to_self = jnp.stack(
            [
                jnp.stack(
                    [
                        self.boreholes[j].h_to_self(time, alpha) if i == j else self.boreholes[j].h_to_borehole(self.boreholes[i], time, alpha)
                        for j in jnp.arange(n_boreholes)
                    ],
                    axis=-2)
                for i in jnp.arange(n_boreholes)
            ],
            axis=1)
        return h_to_self

    def h_to_point(self, p: Array, time: Array, alpha: float, r_min: float = 0.) -> Array:
        n_boreholes = self.n_boreholes
        h_to_point = jnp.stack(
            [
                self.boreholes[j].h_to_point(p, time, alpha, r_min=r_min) for j in jnp.arange(n_boreholes)
                ],
            axis=-2)
        return h_to_point

    @classmethod
    def from_positions(cls, L: ArrayLike, D: ArrayLike, r_b: ArrayLike, x: ArrayLike, y: ArrayLike, basis: Basis, n_segments: int, tilt: float = 0., orientation: float = 0., segment_ratios: ArrayLike | None = None, order: int | None = None) -> Self:
        # Runtime type validation
        if not isinstance(x, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {x}")
        if not isinstance(y, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {y}")
        # Convert input to jax.Array
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        n_boreholes = len(x)
        L = jnp.broadcast_to(L, n_boreholes)
        D = jnp.broadcast_to(D, n_boreholes)
        r_b = jnp.broadcast_to(r_b, n_boreholes)
        tilt = jnp.broadcast_to(tilt, n_boreholes)
        orientation = jnp.broadcast_to(orientation, n_boreholes)
        boreholes = []
        xi = jnp.array([-1., 1.])
        for j in range(n_boreholes):
            p = jnp.array(
                [
                    [x[j], y[j], -D[j]],
                    [x[j] + L[j] * jnp.sin(tilt[j]) * jnp.cos(orientation[j]), y[j] + L[j] * jnp.sin(tilt[j]) * jnp.sin(orientation[j]), -D[j] - L[j] * jnp.cos(tilt[j])],
                ]
            )
            path = Path(xi, p, order=order)
            boreholes.append(Borehole(r_b[j], path, basis, n_segments, segment_ratios=segment_ratios))
        return cls(boreholes)

    @classmethod
    def rectangle_field(cls, N_1: int, N_2: int, B_1: float, B_2: float, L: float, D: float, r_b: float, basis: Basis, n_segments: int, segment_ratios: ArrayLike | None = None, order: int | None = None) -> Self:
        # Borehole positions and orientation
        x = jnp.tile(jnp.arange(N_1), N_2) * B_1
        y = jnp.repeat(jnp.arange(N_2), N_1) * B_2
        return cls.from_positions(L, D, r_b, x, y, basis, n_segments, segment_ratios=segment_ratios)
        
