# -*- coding: utf-8 -*-
from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike
from jax.scipy.special import erfc


@jit
def point_heat_source(p_source: Array, p: Array, time: Array | float, J: Array | float, alpha: float, r_min: float = 0.) -> Array | float:
    """Point heat source solution.

    Parameters
    ----------
    p_source : array or float
        (N, 3,) array of the positions of the point heat sources along
        the trajectory.
    p : array
        (M, 3,) array of the positions at which the point heat source
        solution is evaluated.
    J : array or float
        (N,) array of the norm of the Jacobian of the positions of the
        point heat sources along the trajectory.
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
    # Distance to the mirror point (p')
    r_mirror = jnp.linalg.norm(p_source - p * jnp.array([1, 1, -1]))
    # Point heat source solution
    h = 0.5 * J * erfc(r / jnp.sqrt(4 * alpha * time)) / r - 0.5 * J * erfc(r_mirror / jnp.sqrt(4 * alpha * time)) / r_mirror
    return h
