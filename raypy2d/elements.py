from matplotlib.axes import Axes
from matplotlib.patches import Arc, Circle
import numpy as np

from .utils import rotation_matrix
from .rays import propagate
from . import plotting


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
        r = rotation_matrix(self.theta)
        points = np.dot(r, points.T).T

        # translation
        points = points + self.origin[None, :]

        return points

    def to_element_frame_of_reference(self, rays):
        """
        transform the passed rays to the global frame of reference (e.g. rotated and translated)
        assuming they are in the object frame of reference
        Args:
            points: (numpy.array) two dimensional array with two columns shape (n, 2) where the first
                    column is interpreted as the x coordinate and the second the y coordinate.

        Returns:
            (numpy.array of shape n, 2) with the transformed coordinates
        """

        # transform points
        rays[:, :2] = self.points_to_object_frame_of_reference(rays[:, :2])

        # rotate ray direction
        tan_theta = np.tan(self.theta*np.pi/180.)
        rays[:, 2] = (rays[:, 2] - tan_theta) / (1. + tan_theta * rays[:, 2])

        return rays

    def to_global_frame_of_reference(self, rays):

        # translation
        rays[:, :2] = self.points_to_global_frame_of_reference(rays[:, :2])

        # rotate ray direction
        tan_theta = np.tan(self.theta*np.pi/180.)
        rays[:, 2] = (tan_theta + rays[:, 2]) / (1. - tan_theta * rays[:, 2])

        return rays


class Element(RotateObject):

    def __init__(self, aperture: float, origin=[0., 0.], theta=0.):
        """
        Creates an optical element
        Args:
            aperture: aperture of the element
            origin: position of the center of the element
            theta: rotation angle in degrees of element (with respect the abscissa)
        """
        RotateObject.__init__(self, origin, theta)

        self.aperture = aperture

        self.matrix = np.eye(2)
        self.mirroring = False

    def trace_in_element_frame_of_reference(self, rays):

        # propagation in air
        rays = propagate(rays, 0.)

        rays[:, :2] = self.intersection_of(rays)

        rays = self.transform_rays(rays)

        if self.mirroring:
            rays[:, 2] = -rays[:, 2]

        rays = self.block(rays)

        return rays

    def transform_rays(self, rays):

        # ABCD transformation of element
        rays[:, 1:3] = np.dot(self.matrix, rays[:, 1:3].T).T

        return rays

    def trace(self, rays):

        rays = self.to_element_frame_of_reference(rays)
        rays = self.trace_in_element_frame_of_reference(rays)
        rays = self.to_global_frame_of_reference(rays)

        return rays

    def block(self, rays):
        return rays

    def intersection_of(self, rays):
        return rays[:,:2]

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

    def __init__(self, diameter: float, origin=[0., 0.], theta=0., blocker_diameter: float = float('+Inf')):
        """
        Creates an aperture element
        Args:
            diameter: (float) diameter of aperture
            origin: position of the center of the aperture
            theta: rotation angle of aperture (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
        """

        assert diameter >= 0.
        assert blocker_diameter > diameter
        self.blocker_diameter = blocker_diameter

        Element.__init__(self, diameter, origin, theta)

        self.diameter = self.aperture

    def block(self, rays):
        rays[np.abs(rays[:,1]) > self.diameter / 2., 2] = np.nan
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

    def __init__(self, diameter: float, origin=[0., 0.], theta=0., blocker_diameter: float = float('+Inf')):
        """
        Creates a mirror element
        Args:
            focal_length: (float) focal length of the lens
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of mirror (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
        """

        Aperture.__init__(self, diameter, origin, theta, blocker_diameter)

        self.mirroring = True

    def plot(self, ax: Axes):
        """
        Plots the mirror into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the mirror into

        Returns:
            (tuple) plotted objects
        """

        plotted_objects = Element.plot(self, ax)
        plotted_objects += plotting.plot_aperture(ax, self, **plotting.wall_properties)
        plotted_objects += plotting.plot_blocker(ax, self, self.blocker_diameter)

        return plotted_objects


class Lens(Aperture):

    def __init__(self, focal_length: float, diameter: float, origin=[0., 0.], theta=0.,
                 blocker_diameter: float = float('+Inf')):
        """
        Creates an lens element
        Args:
            focal_length: (float) focal length of the lens
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of aperture (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
        """

        Aperture.__init__(self, diameter, origin, theta, blocker_diameter)

        self.f = focal_length

        self.matrix = np.array([[1., 0],
                                [-1./self.f, 1.]])

        self.draw_arcs = False

    def plot(self, ax: Axes):
        """
        Plots the lens into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the lens into

        Returns:
            (tuple) plotted objects
        """

        plotted_objects = Aperture.plot(self, ax)

        if self.draw_arcs:

            arc_ratio = 0.02
            arc_radius_factor = (0.5*arc_ratio + 0.125 * 1./arc_ratio)
            m = np.array([[self.diameter * (arc_radius_factor - arc_ratio), 0],
                         [-self.diameter * (arc_radius_factor - arc_ratio), 0]])

            m = self.points_to_global_frame_of_reference(m)

            r = arc_radius_factor * self.diameter
            a = 2*np.arctan(2.*arc_ratio)*180./np.pi
            arc = Arc(m[0, :], 2*r, 2*r, self.theta, 180.-a, 180.+a, **plotting.outline_properties.copy())
            ax.add_patch(arc)
            plotted_objects += (arc,)
            arc = Arc(m[1, :], 2*r, 2*r, self.theta, -a, +a, **plotting.outline_properties.copy())
            ax.add_patch(arc)
            plotted_objects += (arc,)

        return plotted_objects


class ParabolicMirror(Aperture):

    def __init__(self, focal_length: float, diameter: float, origin=[0., 0.], theta=0.,
                 blocker_diameter: float = float('+Inf')):
        """
        Creates a parabolic shaped mirror
        Args:
            focal_length: (float) focal length of the lens
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of aperture (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
        """

        Lens.__init__(self, focal_length, diameter, origin, theta, blocker_diameter)

        self.mirroring = True
        self.matrix[1, 0] *= -1

    def intersection_of(self, rays):

        a = rays[:, 2]
        y = rays[:, 1]
        ay = a * y
        x = (2*np.sqrt(self.f*(self.f - ay)) + ay - 2 * self.f) / rays[:, 2]**2

        zero_elements = (a == 0)
        if zero_elements.any():
            x[zero_elements] = y[zero_elements] * y[zero_elements] / (4. * self.f)

        rays[:, 0] = -x
        rays[:, 1] = y - a * x

        return rays[:, :2]

    def plot(self, ax: Axes):
        """
        Plots the parabolic mirror into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the parabolic mirror into

        Returns:
            (tuple) plotted objects
        """

        plotted_objects = Aperture.plot(self, ax)

        y = np.linspace(-self.diameter/2.,self.diameter/2.)

        points = np.stack((1./(4*self.f)*y**2, y)).T

        ticks = plotting.blocker_ticks(y[0], y[-1])
        ticks[:, 0] = 1./(4*self.f)*ticks[:,1]**2

        points = self.points_to_global_frame_of_reference(points)
        ticks = self.points_to_global_frame_of_reference(ticks)

        plotted_objects += plotting.plot_wall(ax, points)
        plotted_objects += plotting.plot_blocker_ticks(ax, ticks)

        return plotted_objects


class DiffractionGrating(Mirror):

    def __init__(self, grating: float, diameter: float, origin=[0., 0.], theta=0.,
                 blocker_diameter: float = float('+Inf'),
                 default_wavelengths: list = [532., 430, 650.]):
        """
        Creates a diffraction grating element
        Args:
            grating: (float) distance of grating in micrometer
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of mirror (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
            default_wavelengths: (list[float], optional) wavelengths in nm, used if rays do not specify
        """

        Mirror.__init__(self, diameter, origin, theta, blocker_diameter)

        self.mirroring = False
        self.grating = grating
        self.default_wavelengths = default_wavelengths

    def transform_rays(self, rays):

        if rays.shape[1] < 4:
            # add a wavelength column
            rays = np.hstack((rays, np.ones((rays.shape[0], 2)) * self.default_wavelengths[0]))
            rays[:, 3] = 0

            for l in self.default_wavelengths[1:]:
                rays_new = rays.copy()
                rays_new[:, 4] = l
                rays = np.vstack((rays, rays_new))

        elif rays.shape[1] < 5:
            rays = np.hstack((rays, np.ones((rays.shape[0], 1)) * self.default_wavelengths[0]))

            for l in self.default_wavelengths[1:]:
                rays_new = rays.copy()
                rays_new[:, 4] = l
                rays = np.vstack((rays, rays_new))

        rays[:, 2] = np.tan(np.arcsin(rays[:, 4]/ self.grating/ 1000. - np.sin(np.arctan(rays[:, 2]))))

        return rays

    def plot(self, ax: Axes):
        """
        Plots the aperture into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the aperture into

        Returns:
            (point, line1, line2) lines plotted
        """

        points = np.array([[0., self.diameter],
                           [0., -self.diameter]]).T / 2.0

        if self.theta != 0.:
            r = rotation_matrix(self.theta)
            points = np.dot(r, points)

        origin = np.array(self.origin)
        points = points + origin[:, None]

        props = { 'color': 'black', 'linewidth': 2}

        lines = ax.plot(origin[0, None], origin[1, None], marker='x', linestyle='', color='black')
        lines += ax.plot(points[0, :], points[1, :], **props)

        return lines