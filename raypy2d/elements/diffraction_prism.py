import numpy as np
from matplotlib.axes import Axes
from enum import Enum
from typing import List

from .. import plotting
from ..rays import Rays, propagate
from .base import Element, MultiElement
from .aperture import Aperture
from .mirror import Mirror


class Glasses(Enum):
    BK7 = 1
    SF11 = 2


class DiffractionPrism(Aperture, MultiElement):

    constants = {
        Glasses.BK7: np.array([[1.03961212, 0.231792344, 1.01046945],
                               [0.00600069867, 0.0200179144, 103.560653]]),
        Glasses.SF11: np.array([[1.73759695, 0.313747346, 1.89878101],
                                [0.013188707, 0.0623068142, 155.23629]])
    }

    def __init__(self, diameter: float, glass: Glasses = Glasses.BK7, origin=[0., 0.], theta=0.,
                 blocker_diameter: float = float('+Inf'),
                 default_wavelengths: list = [532., 430, 650.],
                 flipped: bool = False):
        """
        Creates a diffraction grating element
        Args:
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of mirror (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
            default_wavelengths: (list[float], optional) wavelengths in nm, used if rays do not specify
            flipped: (bool) if the edges should be flipped or not
        """

        Mirror.__init__(self, diameter, origin, theta, blocker_diameter, flipped)

        self.mirroring = False
        self.glass = glass
        self.default_wavelengths = default_wavelengths

        origin2 = self.points_to_global_frame_of_reference(np.array([[np.sqrt(3.), 1]]) * self.aperture/4.)[0, :]

        self.second_interface = Aperture(self.aperture, origin2, theta=theta+60.)

    def refractive_index(self, wavelength: np.array):
        w2 = (wavelength * 1e-3) ** 2
        c1 = self.constants[self.glass][0]
        c2 = self.constants[self.glass][1]
        return np.sqrt(np.sum(c1[:, None] * w2[None, :] / (w2[None, :] - c2[:, None]), axis=0) + 1)

    def transform_rays(self, rays: Rays, out: bool = False):

        index_split = np.isnan(rays.wavelength)
        if np.any(index_split):
            rays.wavelength[index_split] = self.default_wavelengths[0]

            original_rays = rays.array[index_split, :].copy()
            for w in self.default_wavelengths[1:]:
                new_rays = Rays(original_rays)
                new_rays.wavelength = w
                rays.append(new_rays)

        if not out:
            rays.tan_theta = 1. / self.refractive_index(rays.wavelength) * rays.tan_theta
        else:
            rays.tan_theta = self.refractive_index(rays.wavelength) * rays.tan_theta

        return rays

    def trace(self, rays: Rays) -> Rays:
        rays = self.to_element_frame_of_reference(rays)
        rays = self.trace_in_element_frame_of_reference(rays)
        rays = self.to_global_frame_of_reference(rays)

        rays.store()

        rays = self.second_interface.to_element_frame_of_reference(rays)
        rays = propagate(rays, 0.)
        rays = self.transform_rays(rays, out=True)
        rays = self.second_interface.block(rays)
        rays = self.second_interface.to_global_frame_of_reference(rays)

        return rays

    def elements(self) -> List[Element]:
        return [self, self.second_interface]

    def plot(self, ax: Axes):
        """
        Plots the diffraction grating into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the grating into

        Returns:
            (tuple) plotted objects
        """

        plotted_objects = Element.plot(self, ax)
        plotted_objects += Element.plot(self.second_interface, ax)

        points = np.array([[0., self.aperture],
                           [0., -self.aperture],
                           [self.aperture * np.sqrt(3.), 0.]]) / 2.0

        points_transformed = self.points_to_global_frame_of_reference(points)

        # plotted_objects += plotting.plot_wall(ax, points_transformed[0, :], points_transformed[2, :])
        plotted_objects += plotting.plot_wall(ax, points_transformed[1, :], points_transformed[2, :])

        return plotted_objects
