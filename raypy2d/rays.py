import numpy as np
import operator
from matplotlib.axes import Axes

import uuid
from .utils import assure_number_of_columns
from . import plotting


def _view_property(*args, attr='array'):
    def _get(self):
        return operator.getitem(getattr(self, attr), args)

    def _set_x(self, x):
        operator.setitem(getattr(self, attr), args, x)

    return property(_get, _set_x)


class Rays:

    def __init__(self, array: np.array):
        """
        Interprets the passed array as a list of rays
        Args:
            array: (numpy.array) two dimensional with shape of (n, m) where m >= 3. Columns 1, 2 are interpreted as x
                    and y coordinate and column 3 as the propagation angle
        """

        assert len(array.shape) == 2
        assert array.shape[0] >= 1  # minimal one ray
        assert array.shape[1] >= 3  # min. x, y and theta

        # store a view to array
        self.arrays = [assure_number_of_columns(array, 6)]

        # set default direction to forward
        self.forward[np.isnan(self.forward)] = 1.0

    def _get_array(self):
        return self.arrays[-1]

    def _set_array(self, array):
        self.arrays[-1] = array

    array = property(_get_array, _set_array)

    x = _view_property(slice(None), 0)
    y = _view_property(slice(None), 1)
    tan_theta = _view_property(slice(None), 2)
    forward = _view_property(slice(None), 3)
    group = _view_property(slice(None), 4)
    wavelength = _view_property(slice(None), 5)
    points = _view_property(slice(None), slice(None, 2))
    za = _view_property(slice(None), slice(1, 3))

    properties_array = _view_property(slice(None), slice(3, None))

    def copy(self):
        return Rays(self.array.copy())

    def store(self):
        self.arrays.append(self.arrays[-1].copy())
        self.arrays[-2] = self.arrays[-2][:, :3].copy()

    def complete_array(self):

        rows = max([arr.shape[0] for arr in self.arrays])

        arrs = []
        for arr in self.arrays:
            new_rows = rows - arr.shape[0]

            if new_rows > 0:
                arr = np.vstack((arr, np.zeros((new_rows, arr.shape[1]))))
                arr[-new_rows:, :] = np.nan

            arrs.append(arr)

        arrs[-1] = arrs[-1][:, :3]

        tr = np.transpose(np.asarray(arrs), (1, 0, 2))

        return tr

    def traced_array(self):
        return TracedRays(self)

    def plot(self, ax: Axes, **kwargs):

        rs = self.traced_rays()
        rs.plot(ax, **kwargs)


class TracedRays:

    def __init__(self, rays: Rays):

        self.array = rays.complete_array()
        self.properties_array = rays.properties_array.copy()

    x = _view_property(slice(None), slice(None), 0)
    y = _view_property(slice(None), slice(None), 1)
    tan_theta = _view_property(slice(None), slice(None), 2)

    group = _view_property(slice(None), 0, attr='properties_array')
    wavelength = _view_property(slice(None), 1, attr='properties_array')

    points = _view_property(slice(None), slice(None), slice(None, 2))

    def plot(self, ax: Axes, **kwargs):

        props = plotting.ray_properties.copy()
        props.update(kwargs)

        ax.plot(self.x.T, self.y.T, **props)


def propagate(rays: Rays, x: float):
    """
    Propagates the rays in free space up to the x
    Args:
        rays: (Rays) rays to propagate
        x: (float) x-coordinate of the plane up to which the rays should propagate

    Returns:
        (Rays) propagated rays

    """

    dx = x - rays.x

    rays.y = rays.y + rays.tan_theta * dx
    rays.x = x

    # block rays travelling in the opposite direction
    rays.y[np.logical_xor(dx > 0, rays.forward > 0)] = np.nan

    return rays


def point_source_rays(origin=(0., 0.), angle=(-90., 90.), n: int = 9, group: int = None):
    """
    Creates a number of rays from a point source between the specified emission angles
    Args:
        origin: (list, numpy.array of 2 floats) position of the point source
        angle: (list, numpy.array of 2 floats) emission angle
        n: (int) number of rays
        group: (int, None) identifier used for grouping the rays

    Returns:
        (Rays) the created rays
    """

    origin = np.array(origin)

    da = (max(angle) - min(angle)) / float(n-1)
    rays = Rays(np.zeros((n, 4)))
    rays.points = origin[None, :]
    angle = np.arange(0, n) * da + min(angle)
    rays.tan_theta = np.tan(angle * np.pi / 180.)
    m = np.floor(angle/360.)
    angle = (angle - m*360.)
    rays.forward = ((angle < 90.) | (angle > 270.)).astype(float)

    if group is None:
        group = uuid.uuid1().int >> 64

    rays.group = np.array(group).view(dtype=np.float)

    return rays
