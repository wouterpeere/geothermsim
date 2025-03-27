# -*- coding: utf-8 -*-
from typing import List, Tuple

from jax import numpy as jnp
from matplotlib import pyplot as plt

from ._format_axis import _format_axis
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..path import Path


def plot_path(path: Path | List[Path], num: int = 101, view: str = 'all', ax: Axes | Axes3D | None = None) -> Axes | Axes3D | Tuple[Figure, Axes] | Tuple[Figure, Axes3D] | Tuple[Figure, List[Axes | Axes3D]]:
    if view == 'xy' or view == 'yx':
        return _plot_xy(path, num=num, ax=ax)
    if view == 'yz' or view == 'zy':
        return _plot_yz(path, num=num, ax=ax)
    if view == 'xz' or view == 'zx':
        return _plot_xz(path, num=num, ax=ax)
    if view == '3d':
        return _plot_3d(path, num=num, ax=ax)
    if view == 'all':
        fig = plt.figure(layout='constrained')
        axs = [
            fig.add_subplot(221),
            fig.add_subplot(222, projection='3d'),
            fig.add_subplot(223),
            fig.add_subplot(224),
        ]
        axs[0] = _plot_xy(path, num=num, ax=axs[0])
        axs[1] = _plot_3d(path, num=num, ax=axs[1])
        axs[2] = _plot_xz(path, num=num, ax=axs[2])
        axs[3] = _plot_yz(path, num=num, ax=axs[3])
        return fig, axs


def _plot_xy(path: Path | List[Path], num: int = 101, ax: Axes | None = None) -> Axes | Tuple[Figure, Axes]:
    ax_is_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')
    _format_axis(ax, axis_labels=[r'$x$ [m]', r'$y$ [m]'], equal=True)

    xi = jnp.linspace(-1., 1., num=num)
    if isinstance(path, list):
        for _path in path:
            p = _path.f_p(xi)
            ax.plot(p[:, 0], p[:, 1])
        for _path in path:
            p = _path.f_p(xi)
            ax.plot(p[0, 0], p[0, 1], 'k.')
    else:
        p = path.f_p(xi)

        ax.plot(p[:, 0], p[:, 1])
        ax.plot(p[0, 0], p[0, 1], 'k.')

    if ax_is_none:
        return fig, ax
    else:
        return ax


def _plot_yz(path: Path | List[Path], num: int = 101, ax: Axes | None = None) -> Axes | Tuple[Figure, Axes]:
    ax_is_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')
    _format_axis(ax, axis_labels=[r'$y$ [m]', r'$z$ [m]'], equal=True)

    xi = jnp.linspace(-1., 1., num=num)
    if isinstance(path, list):
        for _path in path:
            p = _path.f_p(xi)
            ax.plot(p[:, 1], p[:, 2])
        for _path in path:
            p = _path.f_p(xi)
            ax.plot(p[0, 1], p[0, 2], 'k.')
    else:
        p = path.f_p(xi)

        ax.plot(p[:, 1], p[:, 2])
        ax.plot(p[0, 1], p[0, 2], 'k.')

    if ax_is_none:
        return fig, ax
    else:
        return ax


def _plot_xz(path: Path | List[Path], num: int = 101, ax: Axes | None = None) -> Axes | Tuple[Figure, Axes]:
    ax_is_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')
    _format_axis(ax, axis_labels=[r'$x$ [m]', r'$z$ [m]'], equal=True)

    xi = jnp.linspace(-1., 1., num=num)
    if isinstance(path, list):
        for _path in path:
            p = _path.f_p(xi)
            ax.plot(p[:, 0], p[:, 2])
        for _path in path:
            p = _path.f_p(xi)
            ax.plot(p[0, 0], p[0, 2], 'k.')
    else:
        p = path.f_p(xi)

        ax.plot(p[:, 0], p[:, 2])
        ax.plot(p[0, 0], p[0, 2], 'k.')

    if ax_is_none:
        return fig, ax
    else:
        return ax


def _plot_3d(path: Path | List[Path], num: int = 101, ax: Axes3D | None = None) -> Axes3D | Tuple[Figure, Axes3D]:
    ax_is_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, layout='constrained')
    _format_axis(ax, axis_labels=[r'$x$ [m]', r'$y$ [m]', r'$z$ [m]'], equal=True)

    xi = jnp.linspace(-1., 1., num=num)
    if isinstance(path, list):
        for _path in path:
            p = _path.f_p(xi)
            ax.plot(p[:, 0], p[:, 1], p[:, 2])
        for _path in path:
            p = _path.f_p(xi)
            ax.plot(p[0, 0], p[0, 1], p[0, 2], 'k.')
    else:
        p = path.f_p(xi)

        ax.plot(p[:, 0], p[:, 1], p[:, 2])
        ax.plot(p[0, 0], p[0, 1], p[0, 2], 'k.')

    if ax_is_none:
        return fig, ax
    else:
        return ax
