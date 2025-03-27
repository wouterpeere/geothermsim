# -*- coding: utf-8 -*-
from typing import List, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..borehole.borehole import Borehole
from ..borehole.single_u_tube import SingleUTube
from .plot_path import plot_path


def plot_borehole(borehole: Borehole | SingleUTube, num: int = 101, view: str = 'all', ax: Axes | Axes3D | None = None) -> Axes | Axes3D | Tuple[Figure, Axes] | Tuple[Figure, Axes3D] | Tuple[Figure, List[Axes | Axes3D]]:
    return plot_path(borehole.path, num=num, view=view, ax=ax)
