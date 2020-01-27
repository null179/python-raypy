import numpy as np
from matplotlib.axes import Axes

from .. import plotting
from ..rays import Rays
from .base import Element


class Aperture(Element):

    def __init__(self, diameter: float, origin=[0., 0.], theta=0., blocker_diameter: float = float('+Inf'),
                 flipped=False):
        """
        Creates an aperture element
        Args:
            diameter: (float) diameter of aperture
            origin: position of the center of the aperture
            theta: rotation angle of aperture (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
            flipped: (bool) if the edges should be flipped or not
        """

        assert diameter >= 0.
        assert blocker_diameter > diameter
        self.blocker_diameter = blocker_diameter

        Element.__init__(self, diameter, origin, theta, flipped)

        self.diameter = self.aperture

    def block(self, rays: Rays):
        abs_y = np.abs(rays.y)
        abs_y[np.isnan(abs_y)] = float("+Inf")
        rays.array[abs_y > self.diameter / 2., 2] = np.nan
        return rays

    def plot(self, ax: Axes):
        """
        Plots the aperture into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the aperture into

        Returns:
            (tuple) plotted objects
        """

        plotted_objects = Element.plot(self, ax)
        plotted_objects += plotting.plot_aperture(ax, self)

        plotted_objects += plotting.plot_blocker(ax, self, self.blocker_diameter)

        return plotted_objects
