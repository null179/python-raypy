import numpy as np
from matplotlib.axes import Axes

from .. import plotting
from ..rays import Rays
from . import plot_blockers
from .base import Element
from .lens import Lens

class ParabolicMirror(Lens):

    def __init__(self, focal_length: float, diameter: float, origin=[0., 0.], theta=0.,
                 blocker_diameter: float = float('+Inf'), flipped=False):
        """
        Creates a parabolic shaped mirror
        Args:
            focal_length: (float) focal length of the lens
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of aperture (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
            flipped: (bool) if the edges should be flipped or not
        """

        Lens.__init__(self, focal_length, diameter, origin, theta, blocker_diameter, flipped)
        self.mirroring = True
        self.matrix[1, 1] *= -1.0

        self.x_blocker = 1. / (4 * self.f) * (self.diameter / 2.) ** 2

    def edges(self):
        points = np.array([[self.x_blocker, self.diameter / 2.],
                           [self.x_blocker, -self.diameter / 2.]])

        if self.flipped:
            points = points[::-1, :]

        return self.points_to_global_frame_of_reference(points)

    def intersection_with(self, rays: Rays):
        a = rays.tan_theta
        y = rays.y.copy()
        ay = a * y
        x = (2 * np.sqrt(self.f * (self.f - ay)) + ay - 2 * self.f) / a ** 2

        zero_elements = (a == 0)
        if zero_elements.any():
            x[zero_elements] = y[zero_elements] * y[zero_elements] / (4. * self.f)

        y = rays.y - a * x

        abs_y = np.abs(y)
        abs_y[np.isnan(abs_y)] = float("+Inf")
        I = (abs_y > self.diameter / 2.)
        x[I] = -self.x_blocker
        y[I] = rays.y[I] - a[I] * x[I]

        rays.x = -x
        rays.y = y

        return rays.points

    def plot(self, ax: Axes):
        """
        Plots the parabolic mirror into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the parabolic mirror into

        Returns:
            (tuple) plotted objects
        """

        y = np.linspace(-self.diameter / 2., self.diameter / 2.)

        points = np.stack((1. / (4 * self.f) * y ** 2, y)).T

        points = self.points_to_global_frame_of_reference(points)

        plotted_objects = Element.plot(self, ax)
        plotted_objects += ax.plot(points[:, 0], points[:, 1], **plotting.outline_properties)

        if plot_blockers:
            plotted_objects = plotting.plot_blocker(ax, self, self.blocker_diameter, x=self.x_blocker, width=-0.4)

        return plotted_objects
