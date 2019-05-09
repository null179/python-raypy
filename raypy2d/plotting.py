#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 09.05.2019
# author:  TOS

import numpy as np
from matplotlib.axes import Axes

from .elements import RotateObject

def plot_origin(ax: Axes, origin: np.array, **kwargs):

    props = {'color': 'black', 'linestyle': '', 'marker': 'x'}
    props.update(kwargs)

    return ax.plot(origin[0, None], origin[1, None], **props)

def plot_blocker_ticks(ax: Axes, ticks: np.array, **kwargs):

    props = {'color': 'black'}
    props.update(kwargs)

    return ax.plot(ticks[0, :], ticks[1, :], **kwargs)

def blocker_ticks(y0, y1, dy: float = 1.0, width: float = 0.4):

    tick_points = np.array(np.arange(y0, y1, dy).tolist() + [y1])
    tick_points = np.stack((np.zeros_like(tick_points), np.ones_like(tick_points) * width))

    return tick_points


def blocker_ticks_symmetric(y0, y1, dy: float = 1.0, width: float = 0.4):

    tick_points = blocker_ticks(y0, y1, dy, width)
    tick_points = np.vstack((tick_points, tick_points))

    return tick_points


def plot_blocker(obj: RotateObject, ax: Axes, diameter: float, blocker_diameter: float):
    if blocker_diameter == float('+Inf'):
        blocker_diameter = 2 * diameter
    else:
        blocker_diameter = blocker_diameter

    points = np.array([[0., blocker_diameter],
                       [0., diameter],
                       [0., diameter],
                       [0., -blocker_diameter]]).T / 2.0

    points = obj.to_global_frame_of_reference(points)
    lines = ax.plot(points[0, :2], points[1, :2], **props)
    lines += ax.plot(points[0, 2:], points[1, 2:], **props)