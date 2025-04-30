# -*- coding: utf-8 -*-
from typing import Tuple

from jax import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._format_axis import _format_axis
from ..simulation import gFunction


def plot_gfunction(gfunction: gFunction, ax: Axes | None = None) -> Axes | Tuple[Figure, Axes]:
    """Plot the g-function of a borefield.

    Parameters
    ----------
    gfunction : g-function
        g-Function to be plotted.
    ax : axes, None, default: ``None``
        The axis on which to draw the ground heat exchanger. If `ax` is
        ``None``, a new figure and new axes are created.

    Returns
    -------
    fig : figure
        The figure. Only return if `ax` is ``None``.
    ax : axes
        The axis.

    """
    ax_is_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')
    _format_axis(ax, axis_labels=[r'$\ln(t/t_s)$', r'$g$-function'])

    # Dimensionless time
    time = gfunction.time[1:]
    alpha = gfunction.alpha
    L = gfunction.borefield.L.mean()
    ts = L**2 / (9 * alpha)
    lntts = jnp.log(time / ts)

    # Evaluate g-function if not already evaluated
    try:
        gfunction.g
    except:
        gfunction.simulate()

    # Plot g-function
    ax.plot(lntts, gfunction.g)

    if ax_is_none:
        return fig, ax
    else:
        return ax
