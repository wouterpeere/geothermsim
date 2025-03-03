# -*- coding: utf-8 -*-
from functools import partial

from jax import jit, vmap
from jax import numpy as jnp
import jax

from .load_aggregation import LoadAggregation


class Simulation:

    def __init__(self, borefield, m_flow, cp_f, dt, tmax, T0, alpha, k_s, cells_per_level=5, p=None):
        self.borefield = borefield
        self.m_flow = m_flow
        self.cp_f = cp_f
        self.dt = dt
        self.tmax = tmax
        self.n_times = int(tmax // dt)
        self.T0 = T0
        self.alpha = alpha
        self.k_s = k_s
        self.cells_per_level = cells_per_level
        self.p = p

        self.loadAgg = LoadAggregation(
            borefield, dt, tmax, 1e-6, cells_per_level=cells_per_level, p=p)
        self.initialize_systems_of_equations()

    def initialize_systems_of_equations(self):
        N = self.borefield.n_boreholes * self.borefield.n_nodes
        self.N = N
        self.h_to_self = self.loadAgg.h_to_self[0].reshape((N, N)) / (2 * jnp.pi * self.k_s)
        self.g_in, self.g_b = self.borefield.g_to_self(self.m_flow, self.cp_f)
        self.A = jnp.block(
            [[self.h_to_self, jnp.eye(N), jnp.zeros((N, 1))],
             [-jnp.eye(N), jax.scipy.linalg.block_diag(*[self.g_b[i, :, :] for i in range(self.borefield.n_boreholes)]), self.g_in.reshape((-1, 1))],
             [self.borefield.w.flatten(), jnp.zeros((1, N + 1))]]
            )
        self.B = jnp.zeros(2 * N + 1)

    def simulate(self, Q, Q_small=1e-3):
        self.loadAgg.reset_history()
        time = 0.
        k = 0
        self.q = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        self.T_b = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        self.T_f_in = jnp.zeros(self.n_times)
        while time < self.tmax:
            time = self.loadAgg.next_time_step()
            if callable(Q):
                Q_k = Q(time)
            else:
                Q_k = Q[k]
            T0 = self.T0 - self.loadAgg.temperature()
            self.B = self.B.at[:self.N].set(T0.flatten())
            self.B = self.B.at[-1].set(Q_k)
            X = jnp.linalg.solve(self.A, self.B)
            self.q = self.q.at[k].set(X[:self.N].reshape((self.borefield.n_boreholes, -1)))
            self.T_b = self.T_b.at[k].set(X[self.N:2*self.N].reshape((self.borefield.n_boreholes, -1)))
            self.T_f_in = self.T_f_in.at[k].set(X[-1])
            self.loadAgg.set_current_load(self.q[k] / (2 * jnp.pi * self.k_s))
            k += 1
        return
