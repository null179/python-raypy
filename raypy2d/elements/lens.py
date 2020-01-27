import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Arc

from .. import plotting
from . import plot_blockers
from .base import Element
from .aperture import Aperture


class Lens(Aperture):

    def __init__(self, focal_length: float, diameter: float, origin=[0., 0.], theta=0.,
                 blocker_diameter: float = float('+Inf'), flipped=False):
        """
        Creates an lens element
        Args:
            focal_length: (float) focal length of the lens
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of aperture (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
            flipped: (bool) if the edges should be flipped or not
        """

        Aperture.__init__(self, diameter, origin, theta, blocker_diameter, flipped)

        self.f = focal_length

        self.matrix = np.array([[1., 0],
                                [-1. / self.f, 1.]])

        self.draw_arcs = False

    def plot(self, ax: Axes):
        """
        Plots the lens into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the lens into

        Returns:
            (tuple) plotted objects
        """

        plotted_objects = Element.plot(self, ax)
        plotted_objects += plotting.plot_aperture(ax, self)

        if plot_blockers:
            plotted_objects += plotting.plot_blocker(ax, self, self.blocker_diameter)

        if self.draw_arcs:
            arc_ratio = 0.02
            arc_radius_factor = (0.5 * arc_ratio + 0.125 * 1. / arc_ratio)
            m = np.array([[self.diameter * (arc_radius_factor - arc_ratio), 0],
                          [-self.diameter * (arc_radius_factor - arc_ratio), 0]])

            m = self.points_to_global_frame_of_reference(m)

            r = arc_radius_factor * self.diameter
            a = 2 * np.arctan(2. * arc_ratio) * 180. / np.pi
            arc = Arc(m[0, :], 2 * r, 2 * r, self.theta, 180. - a, 180. + a, **plotting.outline_properties.copy())
            ax.add_patch(arc)
            plotted_objects += (arc,)
            arc = Arc(m[1, :], 2 * r, 2 * r, self.theta, -a, +a, **plotting.outline_properties.copy())
            ax.add_patch(arc)
            plotted_objects += (arc,)

        return plotted_objects
