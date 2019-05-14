from matplotlib.patches import Arrow
from matplotlib.axes import Axes

import numpy as np
from .elements import Element, RotateObject
from .rays import point_source_rays, propagate, Rays
from . import plotting


class Object(RotateObject):

    def __init__(self, height, origin=[0., 0.], theta: float = 0., fans=[0, 0.5, 1.0], n_rays: int = 9):
        """
        Creates an object subject to imaging. The object emits three fans of rays
        Args:
            height: (float) height of the object
            origin: position of the object
            theta: (float) rotation angle in degrees
            fans: (list[float]) position of the ray fans emitted from object
            n_rays: (int) number of rays per fan
        """

        RotateObject.__init__(self, origin, theta)
        self.height = height
        self.fans_at = fans

        self.rays = []
        for i, fan in enumerate(self.fans_at):
            y0 = fan * self.height - self.height / 2.

            rays = point_source_rays([0, y0], [-75, 75], n=n_rays)
            rays = self.to_global_frame_of_reference(rays)

            self.rays.append(rays.array)

        self.rays = Rays(np.vstack(self.rays))

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

    def append(self, element: Element):
        self.elements.append(element)
        self.rays = element.trace(self.rays)
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

