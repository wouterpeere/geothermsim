# -*- coding: utf-8 -*-
from collections.abc import Callable
from functools import partial
from itertools import cycle
from time import perf_counter
from typing import Tuple

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
    T0 : float or callable
        Undisturbed ground temperature (in degree Celcius), or callable
        that takes the time (in seconds) and a 1d array of positions ``z``
        (in meters, negative) as inputs and returns a 1d array of
        undisturbed ground temperatures (in degree Celsius).
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
        Total fluid mass flow rate (in kg/s).

    """

    def __init__(self, borefield: Network, cp_f: float, dt: float, tmax: float, T0: float | Callable[[float, Array], Array | float], alpha: float, k_s: float, cells_per_level: int = 5, p: ArrayLike | None = None, store_node_values: bool = False):
        # Runtime type validation
        if not isinstance(p, ArrayLike) and p is not None:
            raise TypeError(f"Expected arraylike or None input; got {p}")
        # Convert input to jax.Array
        if p is not None:
            p = jnp.asarray(p)

        # --- Class atributes ---
        # Parameters
        self.borefield = borefield
        self.cp_f = cp_f
        self.dt = dt
        self.tmax = tmax
        self.n_nodes = self.borefield.n_boreholes * self.borefield.n_nodes
        self.T0 = T0
        self.alpha = alpha
        self.k_s = k_s
        self.cells_per_level = cells_per_level
        self.p = p
        self.store_node_values = store_node_values
        # Additional attributes
        self.n_times = int(tmax // dt)
        if p is None:
            self.T = None
            self.n_points = 0
        else:
            self.n_points = p.shape[0]

        # --- Initialization ---
        self.loadAgg = LoadAggregation(
            borefield, dt, tmax, alpha, cells_per_level=cells_per_level, p=p)
        self.initialize_system_of_equations()
        # Convert `T0` to callable
        if not callable(T0):
            def undisturbed_ground_temperature(time: float, z: Array | float) -> float:
                return T0
            self.undisturbed_ground_temperature = undisturbed_ground_temperature
        else:
            self.undisturbed_ground_temperature = T0

    def initialize_system_of_equations(self):
        """Initialize the system of equations.

        """
        N = self.n_nodes
        # Thermal response factors
        self.h_to_self = self.loadAgg.h_to_self[0] / (2 * jnp.pi * self.k_s)
        # Initialize system of equations
        self.A = jnp.block(
            [[-jnp.eye(N), jnp.zeros((N, 1))],
             [self.borefield.w.flatten(), jnp.zeros((1, 1))]]
            )
        self.B = jnp.zeros(N + 1)

    def update_system_of_equations(self, m_flow: float | Array, Q: float, T0: Array) -> Tuple[Array, Array]:
        """Update the system of equations.

        Parameters
        ----------
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        Q : float
            Total heat extraction rate (in watts).
        T0 : array
            Borehole wall temperature at nodes (in degree Celsius)
            assuming zero heat extraction rate.

        Returns
        -------
        A : array
            Linear system of equations.
        B : array
            Right-hand side of the linear system of equations.

        """
        N = self.n_nodes
        # Borehole heat transfer rate coefficients for current fluid mass
        # flow rate `m_flow`
        self.g_in, self.g_b = self.borefield.g_to_self(m_flow, self.cp_f)
        # Update system of equation for the current borehole wall
        # temperature `T0` at nodes
        A = self.A.at[:N, :N].set(-(jnp.eye(N) + jnp.einsum('iml,iljn->imjn', self.g_b, self.h_to_self).reshape((N, N))))
        A = A.at[:N, -1].set(self.g_in.flatten())
        B = self.B.at[:N].set(-vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(self.g_b, T0.reshape((self.borefield.n_boreholes, -1))).flatten())
        # Apply total heat extraction rate `Q`
        B = B.at[-1].set(Q)
        return A, B

    def simulate(self, Q: ArrayLike, f_m_flow: Callable[[float], float | Array], m_flow_small: float = 0.01, disp: bool = True, print_every: int = 100):
        """Simulate the borefield.

        Parameters
        ----------
        Q : array_like
            Total heat extraction rate (in watts) at each time step. If
            the array is shorter than the simulation, it is repeated to
            cover the length of the simulation.
        f_m_flow : callable
            Function that takes the total heat extraction rate (in watts)
            as an input and returns the total fluid mas flow rate
            (in kg/s) or an array of fluid mass flow rate per borehole.
        m_flow_small : float, default: ``0.01``
            The minimum fluid mass flow rate (in kg/s). If `f_m_flow`
            returns a smaller total value, the fluid mass flow rate and
            the heat extraction rate are set to zero. If the total value
            is above `m_flow_small`, the minimum fluid mas flow rate per
            borehole is set to `m_flow_small`.
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
            T0 = (
                self.undisturbed_ground_temperature(
                    time, self.borefield.p[..., 2])
                - self.loadAgg.temperature() / (2 * jnp.pi * self.k_s)
            )
            # Current load and fluid mass flow rate
            current_Q = next(Q_cycle)
            self.Q = self.Q.at[k].set(current_Q)
            m_flow = f_m_flow(current_Q)
            if len(jnp.shape(m_flow)) == 0:
                m_flow_network = m_flow
            else:
                m_flow_network = m_flow.sum()
            self.m_flow = self.m_flow.at[k].set(m_flow_network)
            # Only solve if the total fluid mass flow rate is not zero
            if m_flow_network > m_flow_small:
                # Build and solve system of equations
                q, T_b, T_f_in, T_f_out = self._simulate_step(
                    jnp.maximum(m_flow, m_flow_small),
                    current_Q,
                    T0)
                self.T_f_in = self.T_f_in.at[k].set(T_f_in)
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
                self.T = self.T.at[k].set(
                    self.undisturbed_ground_temperature(
                        time, self.p[:, 2])
                    - self.loadAgg.temperature_to_point() / (2 * jnp.pi * self.k_s))
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

    @partial(jit, static_argnames=['self'])
    def _simulate_step(self, m_flow: float | Array, Q: float, T0: Array) -> Tuple[Array, Array, float, float]:
        """Solve a single time step.

        Parameters
        ----------
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or array of fluid mass
            flow rate per borehole.
        Q : float
            The total heat extraction rate (in watts).
        T0 : array
            Borehole wall temperature at nodes assuming zero heat
            extraction rate (in degree Celsius).

        Returns
        -------
        q : array
            The heat extraction rate at the nodes (in W/m).
        T_b : array
            The borehole wall temperature at the nodes (in degree
            Celsius).
        T_f_in : float
            The inlet fluid temperature (in degree Celsius).

        """
        # Build and solve system of equations
        A, B = self.update_system_of_equations(
            m_flow,
            Q,
            T0)
        X = jnp.linalg.solve(A, B)
        # Store results
        q = X[:self.n_nodes].reshape((self.borefield.n_boreholes, -1))
        T_b = T0 - jnp.tensordot(self.h_to_self, q, axes=([-2, -1], [-2, -1]))
        T_f_in = X[-1]
        T_f_out = T_f_in + Q / (jnp.sum(m_flow) * self.cp_f)
        return q, T_b, T_f_in, T_f_out
