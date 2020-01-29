import numpy as np
from matplotlib.axes import Axes

from .. import plotting
from ..rays import Rays
from . import plot_blockers
from .base import Element
from .aperture import Aperture
from .mirror import Mirror

class DiffractionGrating(Aperture):

    def __init__(self, grating: float, diameter: float, origin=[0., 0.], theta=0.,
                 interference=1.,
                 blocker_diameter: float = float('+Inf'),
                 default_wavelengths: list = [532., 430, 650.],
                 flipped: bool = False):
        """
        Creates a diffraction grating element
        Args:
            grating: (float) distance of grating in micrometer
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of mirror (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
            default_wavelengths: (list[float], optional) wavelengths in nm, used if rays do not specify
            flipped: (bool) if the edges should be flipped or not
        """

        Mirror.__init__(self, diameter, origin, theta, blocker_diameter, flipped)

        self.mirroring = False
        self.grating = grating
        self.interference = interference
        self.default_wavelengths = default_wavelengths

    def transform_rays(self, rays: Rays):

        I_split = np.isnan(rays.wavelength)
        if np.any(I_split):
            rays.wavelength[I_split] = self.default_wavelengths[0]

            original_rays = rays.array[I_split, :].copy()
            for w in self.default_wavelengths[1:]:
                new_rays = Rays(original_rays)
                new_rays.wavelength = w
                rays.append(new_rays)

        rays.tan_theta = np.tan(np.arcsin(
            np.sin(np.sin(np.arctan(rays.tan_theta))) - self.interference * rays.wavelength / self.grating / 1000.))

        return rays

    def diffraction_angle_for(self, wavelength: float = 532., theta: float = 0.):
        """
        calculates the diffraction angle for a specific wavelength
        Args:
            wavelength: (float) wavelength of the input ray
            theta: (float) angle of incident (with respect to normal)

        Returns:
            (float) diffraction angle
        """
        return np.arcsin(np.sin(-theta / 180. * np.pi)
                         - self.interference * wavelength / 1000. / self.grating) * 180 / np.pi + theta

    def plot(self, ax: Axes):
        """
        Plots the diffraction grating into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the grating into

        Returns:
            (tuple) plotted objects
        """

        plotted_objects = Element.plot(self, ax)
        plotted_objects += plotting.plot_aperture(ax, self)

        if plot_blockers:
            plotted_objects += plotting.plot_blocker(ax, self, self.blocker_diameter)

        return plotted_objects
