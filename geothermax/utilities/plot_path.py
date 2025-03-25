# -*- coding: utf-8 -*-
from jax import numpy as jnp
import jax
from matplotlib import pyplot as plt

from ._format_axis import _format_axis


def plot_path(path, num=101, view='all', ax=None):
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


def _plot_xy(path, num=101, ax=None):
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


def _plot_yz(path, num=101, ax=None):
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


def _plot_xz(path, num=101, ax=None):
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


def _plot_3d(path, num=101, ax=None):
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
