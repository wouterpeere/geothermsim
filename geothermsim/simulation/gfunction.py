# -*- coding: utf-8 -*-
from functools import partial
from time import perf_counter
from typing import Tuple

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.scipy.linalg import block_diag
from jax.typing import ArrayLike

from ..borefield.network import Network
from .load_history_reconstruction import LoadHistoryReconstruction


class gFunction:
    """g-Function.

    Parameters
    ----------
    borefield: network
        The borefield.
    m_flow : float or array
        Total fluid mass flow rate (in kg/s), or (`n_boreholes`,) array of
        fluid mass flow rate per borehole.
    time : array_like
        Times (in seconds) to evaluate the g-function.
    cp_f : float
        Fluid specific isobaric heat capacity (in J/kg-K).
    alpha : float
        Ground thermal diffusivity (in m^2/s)
    k_s : float
        Ground thermal conductivity (in W/m-K).
    p : array_like or None, default: ``None``
        (`n_points`, 3,) array of positions to evaluate the ground
        temperature. If `p` is ``None``, the ground temperature is not
        evaluated.
    disp : bool, default: ``True``
        Set to ``True`` to print initialization progression messages.

    Attributes
    ----------
    m_flow_network : float
        Total fluid mass flow rate (in kg/s).
    g : array
        The g-function of the borefield.
    q : array
        The heat extraction rate at the nodes (in W/m).
    T_b : array
        The borehole wall temperature at the nodes (in degree Celsius).
    T_f_in : array
        The inlet fluid temperature (in degree Celsius).
    T : array
        The ground temperature (in degree Celsius) at positions `p`.

    """

    def __init__(self, borefield: Network, m_flow: float | Array, cp_f: float, time: ArrayLike, alpha: float, k_s: float, p: ArrayLike | None = None, disp: bool = True):
        # Runtime type validation
        if not isinstance(time, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {time}")
        if not isinstance(p, ArrayLike) and p is not None:
            raise TypeError(f"Expected arraylike or None input; got {p}")
        # Convert input to jax.Array
        time = jnp.asarray(time)
        if p is not None:
            p = jnp.asarray(p)

        # --- Class atributes ---
        # Parameters
        self.borefield = borefield
        self.n_nodes = self.borefield.n_boreholes * self.borefield.n_nodes
        self.m_flow = m_flow
        self.cp_f = cp_f
        self.time = time
        self.time = jnp.concatenate((jnp.array([0.]), self.time))
        self.alpha = alpha
        self.k_s = k_s
        self.p = p
        # Additional attributes
        self.n_times = len(time)
        if p is None:
            self.T = None
            self.n_points = 0
        else:
            self.n_points = p.shape[0]
        if len(jnp.shape(m_flow)) == 0:
            self.m_flow_network = m_flow
        else:
            self.m_flow_network = m_flow.sum()

        # --- Initialization ---
        self.loaHisRec = LoadHistoryReconstruction(
            borefield, time, alpha, p=p, disp=disp)
        self.initialize_systems_of_equations()

    def initialize_systems_of_equations(self):
        """Initialize the system of equations.

        """
        N = self.n_nodes
        # Thermal response factors
        self.h_to_self = jnp.concatenate(
            (
                jnp.zeros(
                    (1,
                     self.borefield.n_boreholes,
                     self.borefield.n_nodes,
                     self.borefield.n_boreholes,
                     self.borefield.n_nodes)
                ),
                self.loaHisRec.h_to_self / (2 * jnp.pi * self.k_s)
            ),
            axis=0)
        # Borehole heat transfer rate coefficients
        self.g_in, self.g_b = self.borefield.g_to_self(self.m_flow, self.cp_f)
        # Initialize system of equations
        self.A = jnp.block(
            [[jnp.eye(N), self.g_in.reshape((-1, 1))],
             [self.borefield.w.flatten(), jnp.zeros((1, 1))]]
            )
        self.B = jnp.zeros(N + 1)
        # Apply constant total heat extraction rate
        self.B = self.B.at[-1].set(2 * jnp.pi * self.k_s * self.borefield.L.sum())

    def update_system_of_equations(self, h_to_self: Array, T0: Array) -> Tuple[Array, Array]:
        """Update the system of equations.

        Parameters
        ----------
        h_to_self : array
            Array of thermal response factors.
        T0 : array
            Borehole wall temperature at nodes assuming zero heat
            extraction rate.

        Returns
        -------
        A : array
            Linear system of equations.
        B : array
            Right-hand side of the linear system of equations.

        """
        N = self.n_nodes
        # Update system of equations for thermal response factors at the
        # current time step `h_to_self`
        A = self.A.at[:N, :N].set(jnp.eye(N) + jnp.einsum('iml,iljn->imjn', self.g_b, h_to_self).reshape((N, N)))
        # Apply current borehole wall temperature at nodes `T0`
        B = self.B.at[:N].set(vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(-self.g_b, T0.reshape((self.borefield.n_boreholes, -1))).flatten())
        return A, B

    def simulate(self, disp: bool = True, print_every: int = 10):
        """Evaluate the g-function.

        Parameters
        ----------
        disp : bool, default: ``True``
            Set to ``True`` to print simulation progression messages.
        print_every : int, default: ``10``
            Period at which simulation progression messages are printed.

        """
        tic = perf_counter()
        next_k = print_every
        if disp:
            print('Simulation start.')
        self.loaHisRec.reset_history()
        # Initialize arrays
        self.q = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        self.T_b = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        self.T_f_in = jnp.zeros(self.n_times)
        if self.p is not None:
            self.T = jnp.zeros((self.n_times, self.n_points))

        # Start simulation
        for k in range(len(self.time) - 1):
            # Advance to next time step
            time = self.loaHisRec.next_time_step()
            dtime = self.time[k + 1] - self.time[k]
            # Temporal and spatial superposition of past loads
            T0 = self.loaHisRec.temperature() / (2 * jnp.pi * self.k_s)
            # Build and solve system of equations
            q, T_b, T_f_in = self._simulate_step(dtime, T0)
            # Apply latest heat extraction rates
            self.loaHisRec.set_current_load(q)
            # Store results
            self.q = self.q.at[k].set(q)
            self.T_b = self.T_b.at[k].set(T_b)
            self.T_f_in = self.T_f_in.at[k].set(T_f_in)
            # Evaluate ground temperatures
            if self.p is not None:
                self.T = self.T.at[k].set(
                    self.loaHisRec.temperature_to_point() / (2 * jnp.pi * self.k_s)
                )
            if k >= next_k:
                next_k += print_every
                toc = perf_counter()
                if disp:
                    print(
                        f'Completed {k} of {self.n_times} time steps. '
                        f'Elapsed time: {toc-tic:.2f} seconds.'
                    )
        T_f_out = self.T_f_in - 2 * jnp.pi * self.k_s * self.borefield.L.sum() / (self.m_flow_network * self.cp_f)
        # Average fluid temperature
        T_f = 0.5 * (self.T_f_in + T_f_out)
        # Borefield thermal resistance
        R_field = self.borefield.effective_borefield_thermal_resistance(self.m_flow, self.cp_f)
        # Effective borehole wall temperature
        self.g = T_f - 2 * jnp.pi * self.k_s * R_field
        if disp:
            toc = perf_counter()
            print(
                f'Simulation end. Elapsed time: {toc-tic:.2f} seconds.'
            )

    @partial(jit, static_argnames=['self'])
    def _simulate_step(self, dtime: float, T0: Array) -> Tuple[Array, Array, float]:
        """Solve a single time step.

        Parameters
        ----------
        dtime : float
            Current time step variation (in seconds).
        T0 : array
            Borehole wall temperature at nodes assuming zero heat
            extraction rate.

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
        # Current thermal response factors
        h_shape = self.h_to_self.shape
        h = vmap(
            lambda _h: jnp.interp(dtime, self.time, _h),
            in_axes=-1,
            out_axes=-1
            )(
            self.h_to_self.reshape((h_shape[0], -1))
        ).reshape(h_shape[1:])
        # Build and solve system of equations
        A, B = self.update_system_of_equations(h, T0)
        X = jnp.linalg.solve(A, B)
        q = X[:self.n_nodes].reshape((self.borefield.n_boreholes, -1))
        T_b = T0 + jnp.tensordot(h, q, axes=([-2, -1], [-2, -1]))
        T_f_in = X[-1]
        return q, T_b, T_f_in
        
