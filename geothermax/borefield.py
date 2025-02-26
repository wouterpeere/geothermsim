# -*- coding: utf-8 -*-
from functools import partial
from itertools import product

from jax import numpy as jnp
import jax

from .borehole import Borehole
from .path import Path


class Borefield:

    def __init__(self, boreholes):
        self.boreholes = boreholes
        self.n_boreholes = len(boreholes)

    def h_to_self(self, time, alpha):
        h = jnp.stack(
            [
                jnp.stack(
                    [
                        self.boreholes[j].h_to_self(time, alpha) if i == j else self.boreholes[j].h_to_borehole(self.boreholes[i], time, alpha)
                        for j in jnp.arange(self.n_boreholes)
                    ],
                    axis=-2)
                for i in jnp.arange(self.n_boreholes)
            ],
            axis=1)
        return h

    def h_to_point(self, p, time, alpha, r_min=0.):
        h = jnp.stack(
            [
                self.boreholes[j].h_to_point(p, time, alpha, r_min=r_min) for j in jnp.arange(self.n_boreholes)
                ],
            axis=-2)
        return h

    @classmethod
    def from_positions(cls, H, D, r_b, x, y, basis, n_segments, tilt=0., orientation=0., segment_ratios=None, order=None):
        n_boreholes = len(x)
        H = jnp.broadcast_to(H, n_boreholes)
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
                    [x[j] + H[j] * jnp.sin(tilt[j]) * jnp.cos(orientation[j]), y[j] + H[j] * jnp.sin(tilt[j]) * jnp.sin(orientation[j]), -D[j] - H[j] * jnp.cos(tilt[j])],
                ]
            )
            path = Path(xi, p, order=order)
            boreholes.append(Borehole(r_b[j], path, basis, n_segments, segment_ratios=segment_ratios))
        return cls(boreholes)

    @classmethod
    def rectangle_field(cls, N_1, N_2, B_1, B_2, H, D, r_b, basis, n_segments, segment_ratios=None, order=None):
        # Borehole positions and orientation
        x = jnp.tile(jnp.arange(N_1), N_2) * B_1
        y = jnp.repeat(jnp.arange(N_2), N_1) * B_2
        return cls.from_positions(H, D, r_b, x, y, basis, n_segments, segment_ratios=segment_ratios)
        
