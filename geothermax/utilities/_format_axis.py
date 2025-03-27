# -*- coding: utf-8 -*-
from typing import List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D


def _format_axis(ax: Axes | Axes3D, axis_labels: List[str] | None = None, equal: bool = False, inverse_y: bool = False) -> Axes | Axes3D:
    ax.tick_params(
        axis='both', which='both', direction='in',
        bottom=True, top=True, left=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        if ax.name == '3d' and len(axis_labels) == 3:
            ax.set_zlabel(axis_labels[2])
    if isinstance(ax, Axes3D):
        ax.zaxis.set_minor_locator(AutoMinorLocator())
    if equal:
        ax.set_aspect('equal', adjustable='datalim')
    if inverse_y:
        ax.yaxis.set_inverted(True)
    return ax
