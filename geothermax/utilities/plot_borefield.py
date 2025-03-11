# -*- coding: utf-8 -*-
from .plot_path import plot_path


def plot_borefield(borefield, num=101, view='all', ax=None):
    paths = [borehole.path for borehole in borefield.boreholes]
    return plot_path(paths, num=num, view=view, ax=ax)
