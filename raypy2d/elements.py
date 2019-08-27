import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Arc

from . import plotting
from .rays import propagate, Rays
from .utils import rotation_matrix

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


class Sensor(Mirror):

    def __init__(self, diameter: float, origin=[0., 0.], theta=0., blocker_diameter: float = float('+Inf'),
                 flipped=False):
        """
        Creates a sensor element
        Args:
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of mirror (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
            flipped: (bool) if the edges should be flipped or not
        """

        Aperture.__init__(self, diameter, origin, theta, blocker_diameter, flipped)
        self.matrix = np.diag([1., 1.])
        self.mirroring = False
