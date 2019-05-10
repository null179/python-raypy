import numpy as np
import operator
from matplotlib.axes import Axes

import uuid
from .utils import assure_number_of_columns
from . import plotting


def _view_property(*args):
    def _get(self):
        return operator.getitem(self.array, args)

    def _set_x(self, x):
        operator.setitem(self.array, args, x)

    return property(_get, _set_x)


class Rays:

    def __init__(self, array: np.array):
        """
        Interprets the passed array as a list of rays
        Args:
            array: (numpy.array) two dimensional with shape of (n, m) where m >= 3. Columns 1, 2 are interpreted as x
                    and y coordinate and column 3 as the tangens of the propagation angle
        """

        assert len(array.shape) == 2
        assert array.shape[0] >= 1  # minimal one ray
        assert array.shape[1] >= 3  # min. x, y and tan_theta

        # store a view to array
        self.arrays = [assure_number_of_columns(array, 5)]

    def _get_array(self):
        return self.arrays[-1]

    def _set_array(self, array):
        self.arrays[-1] = array

    array = property(_get_array, _set_array)

    x = _view_property(slice(None), 0)
    y = _view_property(slice(None), 1)
    tan_theta = _view_property(slice(None), 2)
    group = _view_property(slice(None), 3)
    wavelength = _view_property(slice(None), 4)
    points = _view_property(slice(None), slice(None, 2))
    za = _view_property(slice(None), slice(1, 3))

    def copy(self):
        return Rays(self.array.copy())

    def store(self):
        self.arrays.append(self.array.copy())

    def __add__(self, other):
        """ combines the rays with other rays """
        return np.vstack((self.array, other.array))

    def traced_rays(self):

        rows = max([arr.shape[0] for arr in self.arrays])

        arrs = []
        for arr in self.arrays:
            new_rows = rows - arr.shape[0]

            if new_rows > 0:
                arr = np.vstack((arr, np.zeros((new_rows, arr.shape[1]))))
                arr[-new_rows:, :] = np.nan

            arrs.append(arr)

        tr = np.transpose(np.asarray(arrs), (1, 2, 0))

        return tr

    def to_tracedrays(self):
        return TracedRays(self)

    def plot(self, ax: Axes):

        rs = self.traced_rays()
        ax.plot(rs[:, 0, :], rs[:, 1, 0], **plotting.ray_properties)


class TracedRays:

    def __init__(self, rays: Rays):

        self.array = rays.traced_rays()

    x = _view_property(0, slice(None), slice(None))
    y = _view_property(1, slice(None), slice(None), 1)
    tan_theta = _view_property(2, slice(None), slice(None))
    group = _view_property(3, slice(None), 0)
    wavelength = _view_property(4, slice(None), 0)
    points = _view_property(slice(None, 2), slice(None), slice(None))


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

    rays.y = rays.x + rays.tan_theta * dx
    rays.x = x

    return rays


def point_source_rays(origin=[0., 0.], angle=[-90., 90.], n: int = 9, group: int = None):
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

    da = (max(angle) - min(angle)) / float(n + 1)
    rays = Rays(np.zeros((n, 3)))
    rays.points = origin[None, :]
    rays.tan_theta = np.tan((np.arange(1, n + 1) * da + min(angle)) * np.pi / 180.)

    if group is None:
        group = uuid.uuid1().int >> 64

    rays.group = np.array(group).view(dtype=np.float)

    return rays
