# -*- coding: utf-8 -*-
from jax import numpy as jnp
import jax
from matplotlib import pyplot as plt

from ._format_axis import _format_axis


def plot_basis(basis, num=101, ax=None):
    ax_is_none = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    labels = [f'$\\psi_{n+1}$' for n in range(basis.n_nodes)]
    _format_axis(ax, axis_labels=[r'$\xi$', r'$\psi$'])

    xi = jnp.linspace(-1., 1., num=num)
    psi = basis.f_psi(xi)
    ax.plot(xi, psi, label=labels);
    ax.plot(basis.xi, jnp.ones(basis.n_nodes), 'k.');
    ax.legend()

    plt.tight_layout()

    if ax_is_none:
        return fig, ax
    else:
        return ax
