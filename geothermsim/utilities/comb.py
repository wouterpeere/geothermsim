# -*- coding: utf-8 -*-
from jax import jit
from jax import numpy as jnp
from jax.scipy.special import gammaln


@jit
def comb(N: int, k: int) -> float:
    """Number of combinations (N choose k).

    Parameters
    ----------
    N : int
        Number of things.
    k : int
        Number of elements taken.

    Returns
    -------
    float
        The total number of combinations.

    """
    return jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1))
