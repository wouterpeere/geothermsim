# -*- coding: utf-8 -*-
from functools import partial

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.lax import fori_loop
from jax.typing import ArrayLike

from .utilities import comb


class Multipole:

    def __init__(self, r_b, r_p, p, k_s, k_b, J=3):
        # Runtime type validation
        if not isinstance(r_p, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {r_p}")
        if not isinstance(p, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {p}")
        # Convert input to jax.Array
        r_p = jnp.asarray(r_p)
        p = jnp.atleast_2d(p)

        self.r_b = r_b
        self.r_p = r_p
        self.p = p
        self.k_s = k_s
        self.k_b = k_b
        self.J = J

        self.n_pipes = len(r_p)
        self.sigma = (k_b - k_s) / (k_b + k_s)
        self.z = p[:, 0] + 1.j * p[:, 1]
        self.zz = vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
            )(self.z.conj(), self.z)
        self.dis = vmap(
            jnp.add,
            in_axes=(0, None),
            out_axes=0
            )(self.z, -self.z)

        self.factors()
        self.thermal_resistances_line_zero_beta = self._thermal_resistances_line_zero_beta()
        self.thermal_resistances_multipole = jnp.stack(
            [self._thermal_resistances_multipole(_j) for _j in jnp.arange(1, self.J + 1)],
            axis=-1
        )

    def factors(self):
        k = jnp.arange(1, self.J + 1)
        j = jnp.arange(1, self.J + 1)
        self.factors_line = vmap(
            self._factors_line,
            in_axes=0,
            out_axes=1
        )(k)
        self.factors_multipole = vmap(
            vmap(
                self._factors_multipole,
                in_axes=(0, None),
                out_axes=1
                ),
            in_axes=(None, 0),
            out_axes=-1
        )(k, j)
        self.factors_conjugate_multipole = vmap(
            vmap(
                self._factors_conjugate_multipole,
                in_axes=(0, None),
                out_axes=1
                ),
            in_axes=(None, 0),
            out_axes=-1
        )(k, j)

    @partial(jit, static_argnames=['self'])
    def fluid_temperatures(self, q, T_b, beta):
        a_q = self._thermal_resistances_line(beta)
        a_P = self.thermal_resistances_multipole
        P = self.solve(q, beta)
        T_f = T_b + a_q @ q + jnp.tensordot(a_P, P).real
        return T_f

    def _fluid_temperatures_a_q(self, beta):
        return self.thermal_resistances_zero(beta)

    @partial(jit, static_argnames=['self'])
    def solve(self, q, beta):
        k = jnp.arange(1, self.J + 1)
        beta_k = vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
            )(beta, k)
        coeffs = (1 + beta_k) / (1 - beta_k)
        B = (-self.factors_line @ q).flatten()
        B = jnp.concatenate([B.real, B.imag])
        N = self.n_pipes * self.J
        A = self.factors_multipole.reshape((N, N))
        C = self.factors_conjugate_multipole.reshape((N, N))
        A = jnp.block(
            [[A.real + C.real, -A.imag + C.imag],
             [A.imag + C.imag, A.real - C.real]]
            )
        A = A + jnp.diag(jnp.concatenate([coeffs.flatten(), -coeffs.flatten()]))
        P = jnp.linalg.solve(A, B)
        P = (P[:N] + 1.j * P[N:]).reshape((self.n_pipes, self.J))
        return P

    def _thermal_resistances_line(self, beta):
        R = self.thermal_resistances_line_zero_beta + jnp.diag(beta) / (2 * jnp.pi * self.k_b)
        return R

    def _thermal_resistances_line_zero_beta(self):
        z = self.z
        zz = self.zz
        I = jnp.eye(self.n_pipes)
        absdis = jnp.abs(self.dis)
        R = 1 / (2 * jnp.pi * self.k_b) * (
            jnp.log(self.r_b / ((1 - I) * absdis + jnp.diag(self.r_p)))
            + self.sigma * jnp.log(
                self.r_b**2 / (
                    (1 - I) * jnp.abs(self.r_b**2 - zz)
                    + jnp.diag(self.r_b**2 - jnp.abs(z)**2)
                )
            )
        )
        return R

    def _thermal_resistances_multipole(self, j):
        z = self.z
        zz = self.zz
        I = jnp.eye(self.n_pipes)
        numer_1 = (1 - I) * self.r_p**j
        denom_1 = self.dis**j + I
        numer_2 = vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
            )(z.conj()**j, self.r_p**j)
        denom_2 = (self.r_b**2 - zz)**j
        a_P = (numer_1 / denom_1 + self.sigma * numer_2 / denom_2)
        return a_P

    def _factors_line(self, k):
        z = self.z
        zz = self.zz
        I = jnp.eye(self.n_pipes)
        denom_1 = ((1 - I) * -self.dis + I)**k
        numer_1 = ((1 - I) * self.r_p**k).T
        denom_2 = (self.r_b**2 - zz.T)**k
        numer_2 = self.sigma * vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
            )(self.r_p**k, z.conj()**k)
        a_q = (numer_1 / denom_1 + numer_2 / denom_2) / k / (2 * jnp.pi * self.k_b)
        return a_q

    def _factors_multipole(self, k, j):
        I = jnp.eye(self.n_pipes)
        numerator = vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
            )((-self.r_p)**k, self.r_p**j)
        denominator = self.dis**(k + j) + I
        a_P = (1 - I) * comb(k + j - 1, j - 1) * numerator / denominator
        return a_P

    def _factors_conjugate_multipole(self, k, j):
        j_p_max = jnp.minimum(k, j)
        z = self.z
        a_P_conj = jnp.zeros((self.n_pipes, self.n_pipes), dtype=complex)
        def body_fun(i, val):
            numerator = vmap(
                jnp.multiply,
                in_axes=(0, None),
                out_axes=0
                )(self.r_p**k * z**(j - i), self.r_p**j * z.conj()**(k - i))
            denominator = (self.r_b**2 - self.zz.T)**(k + j - i)
            return val + self.sigma * comb(j, i) * comb(k + j - i - 1, j - 1) * numerator / denominator
        a_P_conj = fori_loop(0, j_p_max + 1, body_fun, a_P_conj)
        return a_P_conj
