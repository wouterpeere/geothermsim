# -*- coding: utf-8 -*-
from typing import List, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..borefield.borefield import Borefield
from ..borefield.network import Network
from .plot_path import plot_path


def plot_borefield(borefield: Borefield | Network, num: int = 101, view: str = 'all', ax: Axes | Axes3D | None = None) -> Axes | Axes3D | Tuple[Figure, Axes] | Tuple[Figure, Axes3D] | Tuple[Figure, List[Axes | Axes3D]]:
    """Plot borefield trajectories

    Parameters
    ----------
    borefield : borefield or network
        Path to be plotted.
    num : int, default: 101
        Number of points along the boreholes.
    view : {'all', 'xy', 'yz', 'xz', '3d'}, default: 'all'
        The view to be plotted. 'all' plots all views on a 2 by 2
        figure.
    ax : axes, axes_3d or None, default: ``None``
        The axis on which to draw the trajectories. Is `ax` is ``None``,
        a new figure and new axes are created.

    Returns
    -------
    fig : figure
        The figure. Only return if `ax` is ``None``.
    ax : axes, axes_3d or list
        The axis.

    """
    paths = [borehole.path for borehole in borefield.boreholes]
    return plot_path(paths, num=num, view=view, ax=ax)
