# -*- coding: utf-8 -*-
from typing import List, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..borefield.borefield import Borefield
from ..borefield.network import Network
from .plot_path import plot_path


def plot_borefield(borefield: Borefield | Network, num: int = 101, view: str = 'all', ax: Axes | Axes3D | None = None) -> Axes | Axes3D | Tuple[Figure, Axes] | Tuple[Figure, Axes3D] | Tuple[Figure, List[Axes | Axes3D]]:
    paths = [borehole.path for borehole in borefield.boreholes]
    return plot_path(paths, num=num, view=view, ax=ax)
