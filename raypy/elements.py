from matplotlib.axes import Axes
import numpy as np

from .utils import rotation_matrix
from .rays import propagate


class Element:

    def __init__(self, origin=[0., 0.], theta=0.):
        """
        Creates an optical element
        Args:
            origin: position of the center of the element
            theta: rotation angle in degrees of element (with respect the abscissa)
        """

        self.origin = np.array(origin)
        self.theta = theta
        self.matrix = np.eye(2)
        self.mirroring = False

    def to_element_frame_of_reference(self, rays):

        # translation
        rays[:,:2] = rays[:,:2] - self.origin[None,:]

        # rotation
        r = rotation_matrix(-self.theta)
        rays[:,:2] = np.dot(r,rays[:,:2].T).T

        # rotate ray direction
        tan_theta = np.tan(self.theta*np.pi/180.)
        rays[:,2] = (rays[:,2] - tan_theta) / (1. + tan_theta * rays[:,2])

        return rays

    def to_global_frame_of_reference(self, rays):

        # rotation
        r = rotation_matrix(self.theta)
        rays[:, :2] = np.dot(r, rays[:, :2].T).T

        # translation
        rays[:, :2] = rays[:, :2] + self.origin[None, :]

        # rotate ray direction
        tan_theta = np.tan(self.theta*np.pi/180.)
        rays[:,2] = (tan_theta + rays[:,2]) / (1. - tan_theta * rays[:,2])

        return rays

    def trace_in_element_frame_of_reference(self, rays):

        # propagation in air
        rays = propagate(rays, 0., -1 if self.mirroring else 1)

        rays[:, :2] = self.intersection_of(rays)

        # ABCD transformation of element
        rays[:, 1:] = np.dot(self.matrix, rays[:, 1:].T).T

        if self.mirroring:
            rays[:, 2] = -rays[:, 2]

        rays = self.block(rays)

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

        self.diameter = diameter
        self.blocker_diameter = blocker_diameter

        Element.__init__(self, origin, theta)

    def block(self, rays):
        rays[np.abs(rays[:,1]) > self.diameter / 2., 2] = np.nan
        return rays

    def plot(self, ax: Axes):
        """
        Plots the aperture into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the aperture into

        Returns:
            (point, line1, line2) lines plotted
        """

        if self.blocker_diameter == float('+Inf'):
            blocker_diameter = 2 * self.diameter
        else:
            blocker_diameter = self.blocker_diameter

        points = np.array([[0., blocker_diameter],
                           [0., self.diameter],
                           [0., -self.diameter],
                           [0., -blocker_diameter]]).T / 2.0


        yticks = np.arange(points[1, 1], points[1, 0]+0.1, 1.0)
        yticks = np.hstack((yticks, -yticks))
        yticks_x = np.stack((np.zeros_like(yticks), np.ones_like(yticks) * 0.4))
        yticks_y = np.stack((yticks, yticks))
        yticks = np.stack((yticks_x, yticks_y))

        if self.theta != 0.:
            r = rotation_matrix(self.theta)
            points = np.dot(r, points)
            yticks[:, 0, :] = np.dot(r, yticks[:, 0, :])
            yticks[:, 1, :] = np.dot(r, yticks[:, 1, :])


        origin = np.array(self.origin)
        points = points + origin[:, None]
        yticks = yticks + origin[:, None, None]

        props = {'color': 'black', 'linewidth': 2}

        lines = ax.plot(origin[0, None], origin[1, None], marker='x', linestyle='', color='black')
        lines += ax.plot(yticks[0, :, :], yticks[1, :, :], color='black')
        lines += ax.plot(points[0, :2], points[1, :2], **props)
        lines += ax.plot(points[0, 2:], points[1, 2:], **props)

        return lines


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
        Plots the aperture into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the aperture into

        Returns:
            (point, line1, line2) lines plotted
        """

        if self.blocker_diameter == float('-Inf'):
            blocker_diameter = 2 * self.diameter
        else:
            blocker_diameter = self.blocker_diameter

        points = np.array([[0., self.diameter],
                           [0., -self.diameter]]).T / 2.0

        yticks = np.arange(points[1, 1], points[1, 0] + 0.1, 1.0)
        yticks_x = np.stack((np.zeros_like(yticks), -np.ones_like(yticks) * 0.4))
        yticks_y = np.stack((yticks, yticks))
        yticks = np.stack((yticks_x, yticks_y))

        if self.theta != 0.:
            r = rotation_matrix(self.theta)
            points = np.dot(r, points)
            yticks[:, 0, :] = np.dot(r, yticks[:, 0, :])
            yticks[:, 1, :] = np.dot(r, yticks[:, 1, :])

        origin = np.array(self.origin)
        points = points + origin[:, None]
        yticks = yticks + origin[:, None, None]

        props = { 'color': 'black', 'linewidth': 2}

        lines = ax.plot(origin[0, None], origin[1, None], marker='x', linestyle='', color='black')
        lines += ax.plot(yticks[0, :, :], yticks[1, :, :], color='black')
        lines += ax.plot(points[0, :], points[1, :], **props)

        return lines


class Lens(Mirror):

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


    def plot(self, ax: Axes):
        """
        Plots the aperture into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the aperture into

        Returns:
            (point, line1, line2) lines plotted
        """

        if self.blocker_diameter == float('-Inf'):
            blocker_diameter = 2 * self.diameter
        else:
            blocker_diameter = self.blocker_diameter

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


class ParabolicMirror(Lens):

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
        self.matrix[1,0] *= -1

    def intersection_of(self, rays):
        x = 2*(np.sqrt(self.f)*np.sqrt(rays[:,2]**2*self.f + rays[:,1]) + rays[:,2] * self.f)
        rays[:, 1] = x
        rays[:, 0] = 1/(4*self.f)*x*x

        return rays[:, :2]

    def plot(self, ax: Axes):
        """
        Plots the parabolic mirror into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the parabolic mirror into

        Returns:
            (point, line1, line2) lines plotted
        """

        if self.blocker_diameter == float('-Inf'):
            blocker_diameter = 2 * self.diameter
        else:
            blocker_diameter = self.blocker_diameter

        y = np.linspace(-self.diameter/2.,self.diameter/2.)
        points = np.stack((1./(4*self.f)*y**2, y))


        yticks = np.arange(y[0], y[-1] + 0.1, 1.0)
        yticks_x = 1./(4*self.f)*yticks**2
        yticks_x = np.stack((yticks_x, yticks_x-np.ones_like(yticks) * 0.4))
        yticks_y = np.stack((yticks, yticks))
        yticks = np.stack((yticks_x, yticks_y))

        if self.theta != 0.:
            r = rotation_matrix(self.theta)
            points = np.dot(r, points)
            yticks[:, 0, :] = np.dot(r, yticks[:, 0, :])
            yticks[:, 1, :] = np.dot(r, yticks[:, 1, :])

        origin = np.array(self.origin)
        points = points + origin[:, None]
        yticks = yticks + origin[:, None, None]

        props = { 'color': 'black', 'linewidth': 2}

        lines = ax.plot(origin[0, None], origin[1, None], marker='x', linestyle='', color='black')
        lines += ax.plot(yticks[0, :, :], yticks[1, :, :], color='black')
        lines += ax.plot(points[0, :], points[1, :], **props)

        return lines


class CurvedMirror(Aperture):

    def __init__(self, focal_length: float, diameter: float, origin=[0., 0.], theta=0.,
                 blocker_diameter: float = float('+Inf')):
        """
        Creates an curved mirror element
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
                                [-1./self.f, 1]])

    def plot(self, ax: Axes):
        """
        Plots the aperture into the passed matplotlib axes
        Args:
            ax: (Axes) the axes to plot the aperture into

        Returns:
            (point, line1, line2) lines plotted
        """

        if self.blocker_diameter == float('-Inf'):
            blocker_diameter = 2 * self.diameter
        else:
            blocker_diameter = self.blocker_diameter

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
