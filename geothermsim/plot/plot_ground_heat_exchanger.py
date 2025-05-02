# -*- coding: utf-8 -*-
from typing import Tuple

from jax import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._format_axis import _format_axis
from ..ground_heat_exchanger import (
CoaxialHeatExchanger,
UTubeHeatExchanger
)


def plot_ground_heat_exchanger(ground_heat_exchanger: CoaxialHeatExchanger | CoaxialHeatExchanger, ax: Axes | None = None) -> Axes | Tuple[Figure, Axes]:
    """Plot a ground heat exchanger.

    Parameters
    ----------
    ground_heat_exchanger : ground heat exchanger
        Ground heat exchanger to be plotted.
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
    _format_axis(ax, axis_labels=[r'$x$ [m]', r'$y$ [m]'], equal=True)

    if isinstance(ground_heat_exchanger, CoaxialHeatExchanger):
        _plot_coaxial_heat_exchanger(ground_heat_exchanger, ax=ax)
    elif isinstance(ground_heat_exchanger, UTubeHeatExchanger):
        _plot_u_tube_heat_exchanger(ground_heat_exchanger, ax=ax)

    if ax_is_none:
        return fig, ax
    else:
        return ax


def _plot_coaxial_heat_exchanger(coaxial_heat_exchanger: CoaxialHeatExchanger, ax: Axes | None = None) -> Axes | Tuple[Figure, Axes]:
    """Plot a U-tube ground heat exchanger.

    Parameters
    ----------
    coaxial_heat_exchanger : coaxial ground heat exchanger
        Ground heat exchanger to be plotted.
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
    _format_axis(ax, axis_labels=[r'$x$ [m]', r'$y$ [m]'], equal=True)

    ghe = coaxial_heat_exchanger

    # Color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Draw dots at min/max x/y for matplotlib to size the figure correctly
    ax.plot(
        [-ghe.r_b, 0., ghe.r_b, 0.],
        [0., ghe.r_b, 0., -ghe.r_b],
        'k.',
        alpha=0.
    )
    # Borehole wall outline
    borehole_wall = plt.Circle(
        (0., 0.),
        radius=ghe.r_b,
        fill=False,
        color='k',
        linestyle='--'
        )
    # Add borehole wall patch
    ax.add_patch(borehole_wall)

    # Pipes
    n_pipes_over_two = ghe._n_pipes_over_two
    for i in range(n_pipes_over_two):
        # Coordinates of pipes
        (x, y) = ghe.p[i]
        # Identify inlet
        if i in ghe.indices_inner:
            dx_inlet = 0.
            dx_outlet = 0.5 * ghe.r_p_out[i] + 0.5 * ghe.r_p_in[i + n_pipes_over_two]
        else:
            dx_inlet = 0.5 * ghe.r_p_in[i] + 0.5 * ghe.r_p_out[i + n_pipes_over_two]
            dx_outlet = 0.

        # Pipe outline (inlet)
        pipe_in_in = plt.Circle(
            (x, y),
            radius=ghe.r_p_in[i],
            fill=False,
            linestyle='-',
            color=colors[i]
            )
        pipe_in_out = plt.Circle(
            (x, y),
            radius=ghe.r_p_out[i],
            fill=False,
            linestyle='-',
            color=colors[i]
            )
        ax.text(
            x + dx_inlet,
            y,
            i,
            ha="center",
            va="center")

        # Pipe outline (outlet)
        pipe_out_in = plt.Circle(
            (x, y),
            radius=ghe.r_p_in[i + n_pipes_over_two],
            fill=False,
            linestyle='-',
            color=colors[i]
            )
        pipe_out_out = plt.Circle(
            (x, y),
            radius=ghe.r_p_out[i + n_pipes_over_two],
            fill=False,
            linestyle='-',
            color=colors[i]
            )
        ax.text(
            x + dx_outlet,
            y,
            i + n_pipes_over_two,
            ha="center",
            va="center")

        # Add pipe patches
        ax.add_patch(pipe_in_in)
        ax.add_patch(pipe_in_out)
        ax.add_patch(pipe_out_in)
        ax.add_patch(pipe_out_out)

    if ax_is_none:
        return fig, ax
    else:
        return ax


def _plot_u_tube_heat_exchanger(u_tube_heat_exchanger: UTubeHeatExchanger, ax: Axes | None = None) -> Axes | Tuple[Figure, Axes]:
    """Plot a U-tube ground heat exchanger.

    Parameters
    ----------
    u_tube_heat_exchanger : U-tube ground heat exchanger
        Ground heat exchanger to be plotted.
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
    _format_axis(ax, axis_labels=[r'$x$ [m]', r'$y$ [m]'], equal=True)

    ghe = u_tube_heat_exchanger

    # Color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Draw dots at min/max x/y for matplotlib to size the figure correctly
    ax.plot(
        [-ghe.r_b, 0., ghe.r_b, 0.],
        [0., ghe.r_b, 0., -ghe.r_b],
        'k.',
        alpha=0.
    )
    # Borehole wall outline
    borehole_wall = plt.Circle(
        (0., 0.),
        radius=ghe.r_b,
        fill=False,
        color='k',
        linestyle='--'
        )
    # Add borehole wall patch
    ax.add_patch(borehole_wall)

    # Pipes
    n_pipes_over_two = ghe._n_pipes_over_two
    for i in range(n_pipes_over_two):
        # Coordinates of pipes
        (x_in, y_in) = ghe.p[i]
        (x_out, y_out) = ghe.p[i + n_pipes_over_two]

        # Pipe outline (inlet)
        pipe_in_in = plt.Circle(
            (x_in, y_in),
            radius=ghe.r_p_in[i],
            fill=False,
            linestyle='-',
            color=colors[i]
            )
        pipe_in_out = plt.Circle(
            (x_in, y_in),
            radius=ghe.r_p_out[i],
            fill=False,
            linestyle='-',
            color=colors[i]
            )
        ax.text(
            x_in,
            y_in,
            i,
            ha="center",
            va="center")

        # Pipe outline (outlet)
        pipe_out_in = plt.Circle(
            (x_out, y_out),
            radius=ghe.r_p_in[i + n_pipes_over_two],
            fill=False,
            linestyle='-',
            color=colors[i]
            )
        pipe_out_out = plt.Circle(
            (x_out, y_out),
            radius=ghe.r_p_out[i + n_pipes_over_two],
            fill=False,
            linestyle='-',
            color=colors[i]
            )
        ax.text(
            x_out,
            y_out,
            i + n_pipes_over_two,
            ha="center",
            va="center")

        # Add pipe patches
        ax.add_patch(pipe_in_in)
        ax.add_patch(pipe_in_out)
        ax.add_patch(pipe_out_in)
        ax.add_patch(pipe_out_out)

    if ax_is_none:
        return fig, ax
    else:
        return ax
