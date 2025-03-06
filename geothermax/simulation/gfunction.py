# -*- coding: utf-8 -*-
from functools import partial

from jax import jit, vmap
from jax import numpy as jnp
import jax

from .load_history_reconstruction import LoadHistoryReconstruction


class gFunction:

    def __init__(self, borefield, m_flow, cp_f, time, alpha, k_s, p=None):
        self.borefield = borefield
        self.m_flow = m_flow
        self.cp_f = cp_f
        self.time = time
        self.n_times = len(time)
        self.time = jnp.concatenate((jnp.array([0.]), self.time))
        self.alpha = alpha
        self.k_s = k_s
        self.p = p
        if p is None:
            self.T = None
            self.n_points = 0
        else:
            self.n_points = p.shape[0]

        self.loaHisRec = LoadHistoryReconstruction(
            borefield, time, alpha, p=p)
        self.initialize_systems_of_equations()

    def initialize_systems_of_equations(self):
        N = self.borefield.n_boreholes * self.borefield.n_nodes
        self.N = N
        self.h_to_self = jnp.concatenate((jnp.zeros((1, N, N)), self.loaHisRec.h_to_self.reshape((-1, N, N)) / (2 * jnp.pi * self.k_s)), axis=0)
        self.g_in, self.g_b = self.borefield.g_to_self(self.m_flow, self.cp_f)
        self.A = jnp.block(
            [[jnp.zeros((N, N)), -jnp.eye(N), jnp.zeros((N, 1))],
             [jnp.eye(N), jax.scipy.linalg.block_diag(*[self.g_b[i, :, :] for i in range(self.borefield.n_boreholes)]), self.g_in.reshape((-1, 1))],
             [self.borefield.w.flatten(), jnp.zeros((1, N + 1))]]
            )
        self.B = jnp.zeros(2 * N + 1)
        self.B = self.B.at[-1].set(2 * jnp.pi * self.k_s * self.borefield.L.sum())

    def simulate(self):
        self.loaHisRec.reset_history()
        self.q = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        self.T_b = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        self.T_f_in = jnp.zeros(self.n_times)
        if self.p is not None:
            self.T = jnp.zeros((self.n_times, self.n_points))
        for k in range(len(self.time) - 1):
            time = self.loaHisRec.next_time_step()
            dtime = self.time[k + 1] - self.time[k]
            h = vmap(
                vmap(
                    lambda _h: jnp.interp(dtime, self.time, _h),
                    in_axes=-1,
                    out_axes=-1),
                in_axes=-1,
                out_axes=-1
                )(self.h_to_self)
            self.A = self.A.at[:self.N, :self.N].set(h)
            T0 = -self.loaHisRec.temperature()
            self.B = self.B.at[:self.N].set(T0.flatten())
            X = jnp.linalg.solve(self.A, self.B)
            self.q = self.q.at[k].set(X[:self.N].reshape((self.borefield.n_boreholes, -1)))
            self.T_b = self.T_b.at[k].set(X[self.N:2*self.N].reshape((self.borefield.n_boreholes, -1)))
            self.T_f_in = self.T_f_in.at[k].set(X[-1])
            self.loaHisRec.set_current_load(self.q[k] / (2 * jnp.pi * self.k_s))
            if self.p is not None:
                self.T = self.T.at[k].set(self.loaHisRec.temperature_to_point())
        T_f_out = self.T_f_in - 2 * jnp.pi * self.k_s * self.borefield.L.sum() / (self.m_flow * self.cp_f)
        # Average fluid temperature
        T_f = 0.5 * (self.T_f_in + T_f_out)
        # Borefield thermal resistance
        R_field = self.borefield.effective_borefield_thermal_resistance(self.m_flow, self.cp_f)
        # Effective borehole wall temperature
        self.g = T_f - 2 * jnp.pi * self.k_s * R_field
        return
