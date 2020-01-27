import numpy as np
from matplotlib.axes import Axes

from .. import plotting
from . import plot_blockers
from .base import Element
from .aperture import Aperture


class Mirror(Aperture):

    def __init__(self, diameter: float, origin=[0., 0.], theta=0., blocker_diameter: float = float('+Inf'),
                 flipped=False):
        """
        Creates a mirror element
        Args:
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of mirror (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
            flipped: (bool) if the edges should be flipped or not
        """

        Aperture.__init__(self, diameter, origin, theta, blocker_diameter, flipped)
        self.matrix = np.diag([1., -1.])
        self.mirroring = True

    def edges(self):
        edges = Aperture.edges(self)
        return edges[::-1, :]

    def plot(self, ax: Axes):
        """
        Plots the mirror into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the mirror into

        Returns:
            (tuple) plotted objects
        """

        plotted_objects = Element.plot(self, ax)
        plotted_objects += plotting.plot_aperture(ax, self, **plotting.outline_properties)

        if plot_blockers:
            plotted_objects += plotting.plot_blocker(ax, self, self.blocker_diameter, width=-0.4)

        return plotted_objects
