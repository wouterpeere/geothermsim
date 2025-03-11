# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def _format_axis(ax, axis_labels=None, equal=False, inverse_y=False):
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
    if ax.name == '3d':
        ax.zaxis.set_minor_locator(AutoMinorLocator())
    if equal:
        ax.set_aspect('equal', adjustable='datalim')
    if inverse_y:
        ax.yaxis.set_inverted(True)
    return ax
