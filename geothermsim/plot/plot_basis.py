# -*- coding: utf-8 -*-
from typing import Tuple

from jax import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..basis import Basis
from ._format_axis import _format_axis


def plot_basis(basis: Basis, num: int = 101, ax: Axes | None = None) -> Axes | Tuple[Figure, Axes]:
    ax_is_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')
    labels = [f'$\\psi_{n+1}$' for n in range(basis.n_nodes)]
    _format_axis(ax, axis_labels=[r'$\xi$', r'$\psi$'])

    xi = jnp.linspace(-1., 1., num=num)
    psi = basis.f_psi(xi)
    ax.plot(xi, psi, label=labels);
    ax.plot(basis.xi, jnp.ones(basis.n_nodes), 'k.');
    ax.legend()

    if ax_is_none:
        return fig, ax
    else:
        return ax
