# -*- coding: utf-8 -*-
from jax import numpy as jnp


def reynolds_number_annular_pipe(m_flow: float, r_p_in: float, r_p_out: float, mu_f: float, rho_f: float) -> float:
    """Reynolds number in an annular pipe.

    Parameters
    ----------
    m_flow : float
        Fluid mass flow rate (in kg/s).
    r_p_in : float
        Outer radius of the inner pipe (in meters).
    r_p_out : float
        Inner radius of the outer pipe (in meters).
    mu_f : float
        Fluid dynamic viscosity (in kg/m-s).
    rho_f : float
        Fluid density (in kg/m3).

    Returns
    -------
    float
        Reynolds number.

    """
    # Hydraulic diameter
    D = 2 * (r_p_out - r_p_in)
    # Fluid velocity
    V_flow = jnp.abs(m_flow) / rho_f
    A_cs = jnp.pi * (r_p_out**2 - r_p_in**2)
    V = V_flow / A_cs
    # Reynolds number
    Re = rho_f * V * D / mu_f
    return Re


def reynolds_number_circular_pipe(m_flow: float, r_p: float, mu_f: float, rho_f: float) -> float:
    """Reynolds number in a circular pipe.

    Parameters
    ----------
    m_flow : float
        Fluid mass flow rate (in kg/s).
    r_p : float
        Inner radius of the pipe (in meters).
    mu_f : float
        Fluid dynamic viscosity (in kg/m-s).
    rho_f : float
        Fluid density (in kg/m3).

    Returns
    -------
    float
        Reynolds number.

    """
    # Hydraulic diameter
    D = 2 * r_p
    # Fluid velocity
    V_flow = jnp.abs(m_flow) / rho_f
    A_cs = jnp.pi * r_p**2
    V = V_flow / A_cs
    # Reynolds number
    Re = rho_f * V * D / mu_f
    return Re
