import numpy as np
from matplotlib.axes import Axes

from .. import plotting
from ..rays import propagate, Rays
from ..utils import rotation_matrix

plot_blockers = True


class RotateObject:

    def __init__(self, origin=[0., 0.], theta=0.):
        """
        Creates an object that is rotated with respect to the original coordinate system
        Args:
            origin: position of the center of the element
            theta: rotation angle in degrees of element (with respect the abscissa)
        """
        self.origin = np.array(origin)
        self.theta = theta

    def points_to_object_frame_of_reference(self, points):
        """
        transform the passed points to the objects frame of reference (e.g. rotated and translated)
        assuming they are in the global frame of reference
        Args:
            points: (numpy.array) two dimensional array with two columns shape (n, 2) where the first
                    column is interpreted as the global x coordinate and the second the global y coordinate.

        Returns:
            (numpy.array of shape n, 2) with the transformed coordinates
        """

        # translation
        points = points - self.origin[None, :]

        # rotation
        if self.theta != 0.:
            r = rotation_matrix(-self.theta)
            points = np.dot(r, points.T).T

        return points

    def points_to_global_frame_of_reference(self, points):
        """
        transform the passed points to the global frame of reference (e.g. rotated and translated)
        assuming they are in the object frame of reference
        Args:
            points: (numpy.array) two dimensional array with two columns shape (n, 2) where the first
                    column is interpreted as the x coordinate and the second the y coordinate.

        Returns:
            (numpy.array of shape n, 2) with the transformed coordinates
        """

        # rotation
        if self.theta != 0.:
            r = rotation_matrix(self.theta)
            points = np.dot(r, points.T).T

        # translation
        points = points + self.origin[None, :]

        return points

    def to_element_frame_of_reference(self, rays: Rays):
        """
        transform the passed rays to the global frame of reference (e.g. rotated and translated)
        assuming they are in the object frame of reference
        Args:
            rays: (numpy.array) two dimensional array with two columns shape (n, 2) where the first
                    column is interpreted as the x coordinate and the second the y coordinate.

        Returns:
            (Rays) with the transformed coordinates
        """

        # transform points
        rays.points = self.points_to_object_frame_of_reference(rays.points)

        # rotate the forward information
        rays_theta = np.arctan(rays.tan_theta) * 180. / np.pi + (rays.forward < 0.) * 180.
        rays_theta = rays_theta - self.theta

        # rotate ray direction
        tan_theta = np.tan(self.theta * np.pi / 180.)
        rays.tan_theta = (rays.tan_theta - tan_theta) / (1. + tan_theta * rays.tan_theta)

        m = np.floor(rays_theta / 360.)
        rays_theta = (rays_theta - m * 360.)

        rays_theta[np.isnan(rays_theta)] = 90.
        rays.forward = ((rays_theta < 90.) | (rays_theta > 270.)).astype(float) - 0.5

        return rays

    def to_global_frame_of_reference(self, rays: Rays):

        # translation
        rays.points = self.points_to_global_frame_of_reference(rays.points)

        # rotate the forward information
        rays_theta = np.arctan(rays.tan_theta) * 180. / np.pi + (rays.forward < 0.) * 180.
        new_theta = rays_theta + self.theta
        new_theta[np.isnan(new_theta)] = 90.
        rays.forward = ((new_theta < 90.) | (new_theta > 270.)).astype(float) - 0.5

        # rotate ray direction
        tan_theta = np.tan(self.theta * np.pi / 180.)
        rays.tan_theta = (tan_theta + rays.tan_theta) / (1. - tan_theta * rays.tan_theta)
        return rays


class Element(RotateObject):

    def __init__(self, aperture: float, origin=[0., 0.], theta=0., flipped=False):
        """
        Creates an optical element
        Args:
            aperture: aperture of the element
            origin: position of the center of the element
            theta: rotation angle in degrees of element (with respect the abscissa)
            flipped: (bool) if the edges should be flipped or not
        """
        RotateObject.__init__(self, origin, theta)

        self.aperture = aperture

        self.matrix = np.eye(2)
        self.mirroring = False

        self.flipped = flipped

    def edges(self):

        points = np.array([[0., -self.aperture],
                           [0., self.aperture]]) / 2.

        if self.flipped:
            points = points[::-1, :]

        return self.points_to_global_frame_of_reference(points)

    def trace_in_element_frame_of_reference(self, rays: Rays) -> Rays:
        # propagation in air
        rays = propagate(rays, 0.)

        rays.points = self.intersection_with(rays)

        rays = self.transform_rays(rays)

        rays = self.block(rays)

        return rays

    def transform_rays(self, rays: Rays) -> Rays:
        # ABCD transformation of element
        rays.za = np.dot(self.matrix, rays.za.T).T

        if self.mirroring:
            rays.forward = -rays.forward

        return rays

    def trace(self, rays: Rays) -> Rays:
        rays = self.to_element_frame_of_reference(rays)
        rays = self.trace_in_element_frame_of_reference(rays)
        rays = self.to_global_frame_of_reference(rays)

        return rays

    def block(self, rays: Rays) -> Rays:
        return rays

    def intersection_with(self, rays: Rays):
        return rays.points

    def plot(self, ax: Axes):
        """
        Plots the element position
        Args:
            ax: (Axes) the axes to plot the element into

        Returns:
            (point, line1, line2) lines plotted
        """

        # plot the origin
        plotted_objects = plotting.plot_origin(ax, self.origin)
        plotted_objects += plotting.plot_aperture(ax, self)

        return plotted_objects