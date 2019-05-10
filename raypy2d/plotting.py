#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 09.05.2019
# author:  TOS

import numpy as np
from matplotlib.axes import Axes

origin_properties = {'color': 'black', 'linestyle': '', 'marker': 'x'}
wall_properties = {'color': 'darkgrey', 'linewidth': 1}
axis_properties = {'color': 'grey', 'linestyle': '-.', 'linewidth': 0.5}
outline_properties = {'color': 'grey', 'linestyle': '-', 'linewidth': 1}
ray_properties = {'linestyle': '-', 'linewidth': 0.5}


def plot_origin(ax: Axes, origin: np.array, **kwargs):

    props = origin_properties.copy()
    props.update(kwargs)

    return ax.plot(origin[0, None], origin[1, None], **props)


def plot_blocker_ticks(ax: Axes, ticks_from: np.array, ticks_to: np.array, **kwargs):

    props = wall_properties.copy()
    props.update(kwargs)

    ticks = np.dstack((ticks_from, ticks_to))

    return ax.plot(ticks[:, 0, :].T, ticks[:, 1, :].T, **props)


def plot_wall(ax: Axes, point_from: np.array, point_to: np.array, **kwargs):

    props = wall_properties.copy()
    props.update(kwargs)

    return ax.plot([point_from[0], point_to[0]], [point_from[1], point_to[1]], **props)


def plot_axis(ax: Axes, points: np.array, **kwargs):

    props = axis_properties.copy()
    props.update(kwargs)

    return ax.plot(points[:, :1], points[:, 1:], **props)


def blocker_ticks(y0, y1, dy: float = 1.0, width: float = 0.4):

    tick_points = np.array(np.arange(y0, y1, dy).tolist() + [y1])
    tick_points_from = np.stack((np.zeros_like(tick_points), tick_points)).T
    tick_points_to = np.stack((np.ones_like(tick_points) * width, tick_points)).T

    return tick_points_from, tick_points_to


def blocker_ticks_symmetric(y0, y1, dy: float = 1.0, width: float = 0.4):

    tick_points_from, tick_points_to = blocker_ticks(y0, y1, dy, width)
    tick_points_from2, tick_points_to2 = blocker_ticks(-y1, -y0, dy, width)

    tick_points_from = np.vstack((tick_points_from, tick_points_from2))
    tick_points_to = np.vstack((tick_points_to, tick_points_to2))

    return tick_points_from, tick_points_to


def default_blocker_diameter(diameter: float, blocker_diameter: float):

    if blocker_diameter == float('+Inf'):
        blocker_diameter = 2 * diameter
    else:
        blocker_diameter = blocker_diameter

    return blocker_diameter


def plot_aperture(ax: Axes, element, **kwargs):

    points = np.array([[0., -element.aperture],
                       [0., element.aperture]]) / 2.0

    points = element.points_to_global_frame_of_reference(points)

    # plot the symmetry axis of the element
    return plot_axis(ax, points, **kwargs)


def plot_blocker(ax: Axes, element, blocker_diameter: float, **kwargs):

    blocker_diameter = default_blocker_diameter(element.aperture, blocker_diameter)

    points = np.array([[0., blocker_diameter],
                       [0., element.aperture],
                       [0., -element.aperture],
                       [0., -blocker_diameter]]) / 2.0

    points = element.points_to_global_frame_of_reference(points)

    # plot the origin
    plotted_objects = plot_origin(ax, element.origin, **kwargs)

    # plot the blocking walls around the aperture
    plotted_objects += plot_wall(ax, points[0, :], points[1, :], **kwargs)
    plotted_objects += plot_wall(ax, points[2, :], points[3, :], **kwargs)

    # plot the blocker ticks
    ticks_from, ticks_to = blocker_ticks_symmetric(element.aperture/2., blocker_diameter/2.)
    ticks_from = element.points_to_global_frame_of_reference(ticks_from)
    ticks_to = element.points_to_global_frame_of_reference(ticks_to)
    plotted_objects += plot_blocker_ticks(ax, ticks_from, ticks_to, **kwargs)

    return plotted_objects
