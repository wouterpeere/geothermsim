# -*- coding: utf-8 -*-
from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike
import numpy as np
from scipy.special import binom


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

        self.factors()
        self.thermal_resistances_line_zero_beta = self._thermal_resistances_line_zero_beta()
        self.thermal_resistances_multipole = np.stack(
            [self._thermal_resistances_multipole(_j) for _j in np.arange(1, self.J + 1)],
            axis=-1
        )

    def factors(self):
        k = np.arange(1, self.J + 1)
        j = np.arange(1, self.J + 1)
        self.factors_line = np.stack(
            [self._factors_line(_k) for _k in k],
            axis=1
        )
        self.factors_multipole = np.stack(
            [
                np.stack(
                    [self._factors_multipole(_k, _j) for _k in k],
                    axis=1
                    )
            for _j in j],
            axis=-1
        )
        self.factors_conjugate_multipole = np.stack(
            [
                np.stack(
                    [self._factors_conjugate_multipole(_k, _j) for _k in k],
                    axis=1
                    )
            for _j in j],
            axis=-1
        )
        # self.factors_line = vmap(
        #     self._factors_line,
        #     in_axes=0,
        #     out_axes=1
        # )(k)
        # self.factors_multipole = vmap(
        #     vmap(
        #         self._factors_multipole,
        #         in_axes=(0, None),
        #         out_axes=1
        #         ),
        #     in_axes=(None, 0),
        #     out_axes=-1
        # )(k, j)
        # self.factors_conjugate_multipole = vmap(
        #     vmap(
        #         self._factors_conjugate_multipole,
        #         in_axes=(0, None),
        #         out_axes=1
        #         ),
        #     in_axes=(None, 0),
        #     out_axes=-1
        # )(k, j)

    def fluid_temperatures(self, q, T_b, beta):
        a_q = self._thermal_resistances_line(beta)
        a_P = self.thermal_resistances_multipole
        P = self.solve_direct(q, beta)
        T_f = T_b + a_q @ q + np.real(np.tensordot(a_P, P))
        return T_f

    def _fluid_temperatures_a_q(self, beta):
        return self.thermal_resistances_zero(beta)

    def solve_direct(self, q, beta):
        k = np.arange(1, self.J + 1)
        coeffs = (1 + np.multiply.outer(beta, k)) / (1 - np.multiply.outer(beta, k))
        B = (-self.factors_line @ q).flatten()
        B = np.concatenate([B.real, B.imag])
        N = self.n_pipes * self.J
        A = self.factors_multipole.reshape((N, N))
        C = self.factors_conjugate_multipole.reshape((N, N))
        A = np.block(
            [[A.real + C.real, -A.imag + C.imag],
             [A.imag + C.imag, A.real - C.real]]
            )
        A = A + np.diag(np.concatenate([coeffs.flatten(), -coeffs.flatten()]))
        P = np.linalg.solve(A, B)
        P = (P[:N] + 1.j * P[N:]).reshape((self.n_pipes, self.J))
        return P

    def solve_iterative(self, q, beta, tol=1e-6):
        k = np.arange(1, self.J + 1)
        coeffs = (1 - np.multiply.outer(beta, k)) / (1 + np.multiply.outer(beta, k))
        P = -coeffs * (self.factors_line @ q)
        diff_0 = np.max(np.abs(P))
        diff = diff_0
        while diff > tol * diff_0:
            P_new = -coeffs * (
                self.factors_line @ q
                + np.tensordot(self.factors_multipole, P)
                + np.tensordot(self.factors_conjugate_multipole, P.conj())
                ).conj()
            diff = np.max(np.abs(P_new - P))
            P = P_new
        return P

    def _thermal_resistances_line(self, beta):
        R = self.thermal_resistances_line_zero_beta + np.diag(beta) / (2 * np.pi * self.k_b)
        return R

    def _thermal_resistances_line_zero_beta(self):
        z = self.z
        I = np.eye(self.n_pipes)
        dis = np.abs(np.add.outer(z, -z))
        R = 1 / (2 * np.pi * self.k_b) * (
            np.log(self.r_b / ((1 - I) * dis + np.diag(self.r_p)))
            + self.sigma * np.log(
                self.r_b**2 / (
                    (1 - I) * np.abs(self.r_b**2 - np.multiply.outer(z.conj(), z))
                    + np.diag(self.r_b**2 - np.abs(z)**2)
                )
            )
        )
        return R

    def _thermal_resistances_multipole(self, j):
        z = self.z
        I = np.eye(self.n_pipes)
        numer_1 = (1 - I) * self.r_p**j
        denom_1 = np.add.outer(z, -z)**j + I
        numer_2 = np.multiply.outer(z.conj()**j, self.r_p**j)
        denom_2 = (self.r_b**2 - np.multiply.outer(z.conj(), z))**j
        a_P = (numer_1 / denom_1 + self.sigma * numer_2 / denom_2)
        return a_P

    def _factors_line(self, k):
        z = self.z
        I = np.eye(self.n_pipes)
        denom_1 = (
            (1 - I) * np.add.outer(-z, z) + I
            )**k
        numer_1 = (
            ((1 - I) * self.r_p**k).T
            )
        denom_2 = (
            self.r_b**2 - np.multiply.outer(z, z.conj())
            )**k
        numer_2 = (
            self.sigma * np.multiply.outer(self.r_p**k, z.conj()**k)
            )
        a_q = (numer_1 / denom_1 + numer_2 / denom_2) / k / (2 * np.pi * self.k_b)
        return a_q

    def _factors_multipole(self, k, j):
        z = self.z
        I = np.eye(self.n_pipes)
        numerator = vmap(
            lambda _r: (-self.r_p)**k * _r**j,
            in_axes=0,
            out_axes=-1
        )(self.r_p)
        denominator = np.add.outer(z, -z)**(k + j) + I
        a_P = (1 - I) * binom(k + j - 1, j - 1) * numerator / denominator
        return a_P

    def _factors_conjugate_multipole(self, k, j):
        j_p_max = np.minimum(k, j)
        z = self.z
        a_P_conj = np.zeros((self.n_pipes, self.n_pipes))
        for j_p in range(j_p_max + 1):
            numerator = np.multiply.outer(self.r_p**k * z**(j - j_p), self.r_p**j * z.conj()**(k - j_p))
            denominator = (self.r_b**2 - np.multiply.outer(z, z.conj()))**(k + j - j_p)
            a_P_conj = a_P_conj + self.sigma * binom(j, j_p) * binom(k + j - j_p - 1, j - 1) * numerator / denominator
        return a_P_conj
