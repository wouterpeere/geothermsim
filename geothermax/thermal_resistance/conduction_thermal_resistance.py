# -*- coding: utf-8 -*-
from jax import numpy as jnp


def conduction_thermal_resistance_circular_pipe(r_p_in: float, r_p_out: float, k_p: float) -> float:
    """Conduction thermal resistance in circular pipe.

    Parameters
    ----------
    r_p_in : float
        Inner radius of the pipe (in meters).
    r_p_out : float
        Outer radius of the pipe (in meters).
    k_p : float
        Pipe thermal conductivity (in W/m-K).

    Returns
    -------
    R_p : float
        Conduction thermal resistance (in m-K/W).

    """
    R_p = jnp.log(r_p_out / r_p_in) / (2 * jnp.pi * k_p)
    return R_p
