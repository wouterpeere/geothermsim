# -*- coding: utf-8 -*-
from collections.abc import Callable
from itertools import cycle
from time import perf_counter

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
    store_node_values : bool, default: ``False``
        Set to True to store the borehole wall temperature and heat
        extraction rates at the nodes.

    Attributes
    ----------
    n_times : int
        Number of time steps.
    q : array
        The heat extraction rate at the nodes (in W/m).
        Only if `store_node_values` if ``True``.
    Q : array
        The total heat extraction rate (in watts).
    T_b : array
        The borehole wall temperature at the nodes (in degree Celsius).
        Only if `store_node_values` if ``True``.
    T_f_in : array
        The inlet fluid temperature (in degree Celsius).
    T_f_out : array
        The outlet fluid temperature (in degree Celsius).
    T : array
        The ground temperature (in degree Celsius) at positions `p`.
    m_flow : float
        Fluid mass flow rate (in kg/s).

    """

    def __init__(self, borefield: Network, cp_f: float, dt: float, tmax: float, T0: float, alpha: float, k_s: float, cells_per_level: int = 5, p: ArrayLike | None = None, store_node_values: bool = False):
        # Runtime type validation
        if not isinstance(p, ArrayLike) and p is not None:
            raise TypeError(f"Expected arraylike or None input; got {p}")
        # Convert input to jax.Array
        if p is not None:
            p = jnp.asarray(p)

        self.borefield = borefield
        self.cp_f = cp_f
        self.dt = dt
        self.tmax = tmax
        self.n_nodes = self.borefield.n_boreholes * self.borefield.n_nodes
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
        self.store_node_values = store_node_values

        self.loadAgg = LoadAggregation(
            borefield, dt, tmax, alpha, cells_per_level=cells_per_level, p=p)
        self.initialize_system_of_equations()

    def initialize_system_of_equations(self):
        """Initialize the system of equations.

        """
        N = self.n_nodes
        self.h_to_self = self.loadAgg.h_to_self[0] / (2 * jnp.pi * self.k_s)
        self.A = jnp.block(
            [[-jnp.eye(N), jnp.zeros((N, 1))],
             [self.borefield.w.flatten(), jnp.zeros((1, 1))]]
            )
        self.B = jnp.zeros(N + 1)

    def update_system_of_equations(self, m_flow: float, Q: float, T0: Array):
        """Update the system of equations.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        Q : float
            Total heat extraction rate (in watts).
        T0 : array
            Borehole wall temperature at nodes (in degree Celsius)
            assuming zero heat extraction rate.

        """
        N = self.n_nodes
        self.g_in, self.g_b = self.borefield.g_to_self(m_flow, self.cp_f)
        self.A = self.A.at[:N, :N].set(-(jnp.eye(N) + jnp.einsum('iml,iljn->imjn', self.g_b, self.h_to_self).reshape((N, N))))
        self.A = self.A.at[:N, -1].set(self.g_in.flatten())
        self.B = self.B.at[:N].set(-vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(self.g_b, T0.reshape((self.borefield.n_boreholes, -1))).flatten())
        self.B = self.B.at[-1].set(Q)

    def simulate(self, Q: ArrayLike, f_m_flow: Callable[[float], float], m_flow_small: float = 0.01, disp: bool = True, print_every: int = 100):
        """Simulate the borefield.

        Parameters
        ----------
        Q : array_like
            Total heat extraction rate (in watts) at each time step. If
            the array is shorter than the simulation, it is repeated to
            cover the length of the simulation.
        f_m_flow : callable
            Function that takes the total heat extraction rate (in watts)
            as an input and returns the fluid mas flow rate (in kg/s).
        m_flow_small : float, default: ``0.01``
            The minimum fluid mass flow rate (in kg/s). If `f_m_flow`
            returns a smaller value, the fluid mass flow rate and the heat
            extraction rate are set to zero.
        disp : bool, default: ``True``
            Set to ``True`` to print simulation progression messages.
        print_every : int, default: ``100``
            Period at which simulation progression messages are printed.

        """
        tic = perf_counter()
        next_k = print_every
        if disp:
            print('Simulation start.')
        self.loadAgg.reset_history()
        time = 0.
        k = 0
        Q_cycle = cycle(Q)
        # Initialize arrays
        self.T_f_in = jnp.zeros(self.n_times)
        self.T_f_out = jnp.zeros(self.n_times)
        self.Q = jnp.zeros(self.n_times)
        self.m_flow = jnp.zeros(self.n_times)
        if self.store_node_values:
            self.q = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
            self.T_b = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        if self.p is not None:
            self.T = jnp.zeros((self.n_times, self.n_points))

        # Start simulation
        while time < self.tmax:
            # Advance to next time step
            time = self.loadAgg.next_time_step()
            # Temporal and spatial superposition of past loads
            T0 = self.T0 - self.loadAgg.temperature() / (2 * jnp.pi * self.k_s)
            # Current load and fluid mass flow rate
            Q_k = next(Q_cycle)
            self.Q = self.Q.at[k].set(Q_k)
            m_flow = f_m_flow(Q_k)
            self.m_flow = self.m_flow.at[k].set(m_flow)
            # Only solve if the fluid mass flow rate is not zero
            if m_flow > m_flow_small:
                # Build and solve system of equations
                self.update_system_of_equations(m_flow, Q_k, T0)
                X = jnp.linalg.solve(self.A, self.B)
                # Store results
                q = X[:self.n_nodes].reshape((self.borefield.n_boreholes, -1))
                T_b = T0 - jnp.tensordot(self.h_to_self, q, axes=([-2, -1], [-2, -1]))
                T_f_in = X[-1]
                self.T_f_in = self.T_f_in.at[k].set(T_f_in)
                T_f_out = T_f_in + Q_k / (m_flow * self.cp_f)
                self.T_f_out = self.T_f_out.at[k].set(T_f_out)
                # Apply latest heat extraction rates
                self.loadAgg.set_current_load(q)
                if self.store_node_values:
                    self.q = self.q.at[k].set(q)
                    self.T_b = self.T_b.at[k].set(T_b)
            else:
                # If the fluid mass flow rate is zero,
                # set fluid temperatures to nan
                self.Q = self.Q.at[k].set(0.)
                self.T_f_in = self.T_f_in.at[k].set(jnp.nan)
                self.T_f_out = self.T_f_out.at[k].set(jnp.nan)
                if self.store_node_values:
                    self.T_b = self.T_b.at[k].set(T0)
            # Evaluate ground temperatures
            if self.p is not None:
                self.T = self.T.at[k].set(self.T0 - self.loadAgg.temperature_to_point() / (2 * jnp.pi * self.k_s))
            k += 1
            if k >= next_k:
                next_k += print_every
                toc = perf_counter()
                if disp:
                    print(
                        f'Completed {k} of {self.n_times} time steps. '
                        f'Elapsed time: {toc-tic:.2f} seconds.'
                    )
        if disp:
            print(
                f'Simulation end. Elapsed time: {toc-tic:.2f} seconds.'
            )
