from matplotlib.patches import Arrow
from matplotlib.axes import Axes

import numpy as np
from .elements import Element, RotateObject, Sensor
from .utils import place_relative_to
from .rays import point_source_rays, propagate, Rays
from . import plotting
from typing import List


class Object(RotateObject):

    def __init__(self, height, origin=[0., 0.], theta: float = 0., fans=[0, 0.5, 1.0], n: int = 9, angle=[-75, 75]):
        """
        Creates an object subject to imaging. The object emits three fans of rays
        Args:
            height: (float) height of the object
            origin: position of the object
            theta: (float) rotation angle in degrees
            fans: (list[float]) position of the ray fans emitted from object
            n: (int) number of rays per fan
            angle: (list[float]) default emission angles for ray fans
        """

        RotateObject.__init__(self, [0, 0], theta)
        self.height = height
        self.fans_at = fans

        self.rays = []
        for i, fan in enumerate(self.fans_at):
            y0 = fan * self.height - self.height / 2.

            # backwards compatibility
            if isinstance(angle, int):
                angle = [-angle, angle]

            rays = point_source_rays([0, y0], angle=angle, n=n)
            rays = self.to_global_frame_of_reference(rays)

            self.rays.append(rays.array)

        self.rays = Rays(np.vstack(self.rays))

        # set correct origin
        self.origin = np.array(origin)

        # transform
        self.rays = self.to_global_frame_of_reference(self.rays)

    def edges(self):
        points = np.array([[0, -self.height],
                           [0, self.height]]) / 2.

        return self.points_to_global_frame_of_reference(points)

    def plot(self, ax):

        points = self.edges()

        arrow = Arrow(points[1, 0], points[1, 1],
                      dx=points[0, 0] - points[1, 0],
                      dy=points[0, 1] - points[1, 1],
                      color='blue')

        ax.add_patch(arrow)

        return [arrow]


class OpticalPath:

    def __init__(self, obj: Object = None, **kwargs):

        self.elements = []
        self.obj = obj
        if self.obj is None:
            self.rays = point_source_rays(**kwargs)
        else:
            self.rays = obj.rays

        self.rays.store()

        self.sensors = []

    def append(self, *elements: List[Element], distance=0., theta=0.):
        """
        Append an elements to the path at an optional distance relative to the previous element
        Args:
            elements: list(Element) elements to append
            distance: (float, optional) distance to previous element
            theta: (float, optional) angle for the distance to previous element
        """
        if len(self.elements) > 0:
            ref_element = self.elements[-1]
        else:
            ref_element = RotateObject()
        for element in elements:
            if not (distance == 0. and theta == 0.):
                place_relative_to(ref_element, element, distance, theta)
            self.elements.append(element)

            if isinstance(element, Sensor):
                self.sensors.append(element)
            element.trace(self.rays)
            self.rays.store()

    def propagate(self, x):
        self.rays = propagate(self.rays, x)
        self.rays.store()

    def plot(self, ax: Axes):

        plotted_objects = []

        # plot all elements
        for i, element in enumerate(self.elements):
            plotted_objects += element.plot(ax)

            if i > 0:
                plotted_objects += plotting.plot_maximal_aperture(ax, self.elements[i-1], element)

            elif self.obj is not None:
                plotted_objects += plotting.plot_maximal_aperture(ax, self.obj, element)

        # plot object
        if self.obj is not None:
            plotted_objects += self.obj.plot(ax)

        # plot rays
        plotted_objects += self.rays.plot(ax)

        ax.relim(visible_only=True)
        ax.autoscale_view()

        return plotted_objects