# -*- coding: utf-8 -*-
from jax import jit
from jax import numpy as jnp
from jax.scipy.special import gammaln


@jit
def comb(N, k):
    return jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1))
