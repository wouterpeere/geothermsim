# -*- coding: utf-8 -*-
from collections.abc import Callable

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.scipy.linalg import block_diag
from jax.typing import ArrayLike

from ..borefield.network import Network
from .load_aggregation import LoadAggregation


class Simulation:
    """Simulation.

    Parameters
    ----------
    borefield: network
        The borefield.
    m_flow : float
        Fluid mass flow rate (in kg/s).
    dt : float
        Time step (in seconds).
    tmax : float
        Maximum time of the simulation (in seconds).
    T0 : float
        Undisturbed ground temperature (in degree Celcius).
    cp_f : float
        Fluid specific isobaric heat capacity (in J/kg-K).
    alpha : float
        Ground thermal diffusivity (in m^2/s)
    k_s : float
        Ground thermal conductivity (in W/m-K).
    cells_per_level : int, default: ``5``
        Number of cells per aggregation level.
    p : array_like or None, default: ``None``
        (`n_points`, 3,) array of positions to evaluate the ground
        temperature. If `p` is ``None``, the ground temperature is not
        evaluated.

    Attributes
    ----------
    n_times : int
        Number of time steps.
    q : array
        The heat extraction rate at the nodes (in W/m).
    T_b : array
        The borehole wall temperature at the nodes (in degree Celsius).
    T_f_in : array
        The inlet fluid temperature (in degree Celsius).
    T : array
        The ground temperature (in degree Celsius) at positions `p`.

    """

    def __init__(self, borefield: Network, m_flow: float, cp_f: float, dt: float, tmax: float, T0: float, alpha: float, k_s: float, cells_per_level: int = 5, p: ArrayLike | None = None):
        # Runtime type validation
        if not isinstance(p, ArrayLike) and p is not None:
            raise TypeError(f"Expected arraylike or None input; got {p}")
        # Convert input to jax.Array
        if p is not None:
            p = jnp.asarray(p)

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
        if p is None:
            self.T = None
            self.n_points = 0
        else:
            self.n_points = p.shape[0]

        self.loadAgg = LoadAggregation(
            borefield, dt, tmax, alpha, cells_per_level=cells_per_level, p=p)
        self.initialize_systems_of_equations()

    def initialize_systems_of_equations(self):
        """Initialize the system of equations.

        """
        N = self.borefield.n_boreholes * self.borefield.n_nodes
        self.N = N
        self.h_to_self = self.loadAgg.h_to_self[0].reshape((N, N)) / (2 * jnp.pi * self.k_s)
        self.g_in, self.g_b = self.borefield.g_to_self(self.m_flow, self.cp_f)
        self.A = jnp.block(
            [[self.h_to_self, jnp.eye(N), jnp.zeros((N, 1))],
             [-jnp.eye(N), block_diag(*[self.g_b[i, :, :] for i in range(self.borefield.n_boreholes)]), self.g_in.reshape((-1, 1))],
             [self.borefield.w.flatten(), jnp.zeros((1, N + 1))]]
            )
        self.B = jnp.zeros(2 * N + 1)

    def simulate(self, Q: Callable[[float], float]):
        """Evaluate the g-function.

        Parameters
        ----------
        Q : callable
            Total heat extraction rate (in watts) as a function of time
            (in seconds).
        """
        self.loadAgg.reset_history()
        time = 0.
        k = 0
        self.q = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        self.T_b = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        self.T_f_in = jnp.zeros(self.n_times)
        if self.p is not None:
            self.T = jnp.zeros((self.n_times, self.n_points))
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
            if self.p is not None:
                self.T = self.T.at[k].set(self.T0 - self.loadAgg.temperature_to_point())
            k += 1
