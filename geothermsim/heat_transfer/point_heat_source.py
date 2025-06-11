# -*- coding: utf-8 -*-
from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike
from jax.scipy.special import erfc


def point_heat_source(p_source: Array, p: Array, time: Array | float, alpha: float, r_min: float = 0.) -> Array | float:
    """Point heat source solution.

    Parameters
    ----------
    p_source : array or float
        (N, 3,) array of the positions of the point heat sources along
        the trajectory.
    p : array
        (M, 3,) array of the positions at which the point heat source
        solution is evaluated.
    time : array or float
        (K,) array of times (in seconds).
    alpha : float
        Ground thermal diffusivity (in m^2/s).
    r_min : float, default: ``0.``
        Minimum distance (in meters) between point heat sources and
        positions `p`.

    Returns
    -------
    array or float
        (K, M, N,) array of values of the point heat source solution.
        For each of the parameters `xi`, `p` and `time`, the
        corresponding axis is removed if the parameter is supplied as
        a ``float``.

    """
    if len(jnp.shape(time)) > 0:
        return vmap(
            point_heat_source,
            in_axes=(None, None, -1, None, None, None)
        )(p_source, p, time, J, alpha, r_min)
    if len(jnp.shape(p)) > 1:
        return vmap(
            point_heat_source,
            in_axes=(None, -2, None, None, None, None)
        )(p_source, p, time, J, alpha, r_min)
    if len(jnp.shape(p_source)) > 1:
        return vmap(
            point_heat_source,
            in_axes=(-2, None, None, 0, None, None)
        )(p_source, p, time, J, alpha, r_min)
    # Distance to the real point (p)
    r = jnp.sqrt(((p_source - p)**2).sum() + r_min**2)
    return _point_heat_source(r, time, alpha)


@jit
def _point_heat_source(r: float, time: float, alpha: float) -> float:
    """Point heat source solution.

    Parameters
    ----------
    r : float
        Distance between the point source and the evaluation point (in
        meters).
    time : float
        Time (in seconds).
    alpha : float
        Ground thermal diffusivity (in m^2/s).

    Returns
    -------
    float
        Point heat source solution.

    """
    # Point heat source solution
    h = 0.5 * erfc(r / jnp.sqrt(4 * alpha * time)) / r
    return h
