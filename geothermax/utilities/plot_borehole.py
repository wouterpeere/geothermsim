# -*- coding: utf-8 -*-
from .plot_path import plot_path


def plot_borehole(borehole, num=101, view='all', ax=None):
    return plot_path(borehole.path, num=num, view=view, ax=ax)
