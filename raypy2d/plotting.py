#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 09.05.2019
# author:  TOS

import numpy as np
from matplotlib.axes import Axes

origin_properties = {'color': 'black', 'linestyle': '', 'marker': 'x'}
wall_properties = {'color': 'darkgrey', 'linewidth': 1}
axis_properties = {'color': 'grey', 'linestyle': '-.', 'linewidth': 1}
outline_properties = {'color': 'grey', 'linestyle': '-', 'linewidth': 1}


def plot_origin(ax: Axes, origin: np.array, **kwargs):

    props = origin_properties.copy()
    props.update(kwargs)

    return ax.plot(origin[0, None], origin[1, None], **props)


def plot_blocker_ticks(ax: Axes, ticks: np.array, **kwargs):

    props = wall_properties.copy()
    props.update(kwargs)

    return ax.plot(ticks[0, :, :], ticks[1, :, :], **kwargs)


def plot_wall(ax: Axes, points: np.array, **kwargs):

    props = wall_properties.copy()
    props.update(kwargs)

    return ax.plot(points[0, :], points[1, :], **kwargs)


def plot_axis(ax: Axes, points: np.array, **kwargs):

    props = axis_properties.copy()
    props.update(kwargs)

    return ax.plot(points[0, :], points[1, :], **kwargs)


def blocker_ticks(y0, y1, dy: float = 1.0, width: float = 0.4):

    tick_points = np.array(np.arange(y0, y1, dy).tolist() + [y1])
    tick_points = np.stack((np.zeros_like(tick_points), np.ones_like(tick_points) * width))

    return tick_points


def blocker_ticks_symmetric(y0, y1, dy: float = 1.0, width: float = 0.4):

    tick_points = blocker_ticks(y0, y1, dy, width)
    tick_points = np.vstack((tick_points, tick_points))

    return tick_points


def default_blocker_diameter(diameter: float, blocker_diameter: float):

    if blocker_diameter == float('+Inf'):
        blocker_diameter = 2 * diameter
    else:
        blocker_diameter = blocker_diameter

    return blocker_diameter


def plot_aperture(ax: Axes, element, **kwargs):

    points = np.array([[0., element.aperture],
                       [0., element.aperture]]).T / 2.0

    points = element.points_to_global_frame_of_reference(points)

    # plot the symmetry axis of the element
    return plot_axis(ax, points, **kwargs)


def plot_blocker(ax: Axes, element, blocker_diameter: float, **kwargs):

    blocker_diameter = default_blocker_diameter(element.aperture, blocker_diameter)

    points = np.array([[0., blocker_diameter],
                       [0., element.aperture],
                       [0., -element.aperture],
                       [0., -blocker_diameter]]).T / 2.0

    points = element.points_to_global_frame_of_reference(points)

    # plot the origin
    plotted_objects = plot_origin(ax, element.origin, **kwargs)

    # plot the blocking walls around the aperture
    plotted_objects += plot_wall(ax, points[:, :2], **kwargs)
    plotted_objects += plot_wall(ax, points[:, 2:], **kwargs)

    # plot the blocker ticks
    ticks = blocker_ticks_symmetric(element.aperture, blocker_diameter)
    ticks = element.points_to_global_frame_of_reference(ticks)
    plotted_objects += plot_blocker_ticks(ax, ticks, **kwargs)

    return plotted_objects
