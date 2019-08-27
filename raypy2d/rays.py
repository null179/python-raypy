import numpy as np
import operator
from itertools import cycle
from matplotlib.axes import Axes
from matplotlib import rcParams

import uuid
from .utils import assure_number_of_columns, wavelength_to_rgb, rolling_window
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

    properties_array = _view_property(slice(None), slice(4, None))

    @property
    def n(self):
        return self.array.shape[0]

    def copy(self):
        return Rays(self.array.copy())

    def store(self):
        self.arrays.append(self.arrays[-1])
        self.arrays[-2] = self.arrays[-2][:, :3].copy()

    def append(self, rays):
        self.array = np.vstack((self.array, rays.array))

    def complete_array(self):

        rows = max([arr.shape[0] for arr in self.arrays])

        arrays = []
        for arr in self.arrays:
            new_rows = rows - arr.shape[0]

            if new_rows > 0:
                arr = np.vstack((arr, np.zeros((new_rows, arr.shape[1]))))
                arr[-new_rows:, :] = np.nan

            arrays.append(arr)

        arrays[-1] = arrays[-1][:, :3]

        tr = np.transpose(np.asarray(arrays), (1, 0, 2))

        return tr

    def ray_crossings(self, element=None):
        return RayCrossings.from_traced_rays(self.traced_rays(), element)

    def traced_rays(self):
        return TracedRays.from_rays(self)

    def plot(self, ax: Axes, **kwargs):

        rs = self.traced_rays()
        return rs.plot(ax, **kwargs)


class TracedRays:

    @staticmethod
    def from_rays(rays: Rays):
        return TracedRays(rays.complete_array(), rays.properties_array.copy())

    def __init__(self, array: np.array, properties_array: np.array):
        self.array = array
        self.properties_array = properties_array

    @property
    def n(self):
        return self.array.shape[0]

    x = _view_property(slice(None), slice(None), 0)
    y = _view_property(slice(None), slice(None), 1)
    tan_theta = _view_property(slice(None), slice(None), 2)

    group = _view_property(slice(None), 0, attr='properties_array')
    wavelength = _view_property(slice(None), 1, attr='properties_array')

    points = _view_property(slice(None), slice(None), slice(None, 2))

    def __getitem__(self, item):
        ix, iy = item
        return TracedRays(self.array[ix, iy, :],
                          self.properties_array[ix, :])

    def ray_crossings(self, element=None):
        """
        Calculate all crossings of the rays and returns the crossings and the properties of the two involved rays
        crossings (n_elements - 1,
        Args:
            element: (int, optional) element
        Returns:
            crossings, properties1, properties2 (np.array, np.array, np.array)
        """

        if element is not None:
            i_valid = np.any(~np.isnan(self.points[:, element, :]), axis=1)
            array = self.array[i_valid, :, :2]
            properties_array = self.properties_array[i_valid, :]
        else:
            array = self.array[:, :, :2]
            properties_array = self.properties_array

        d = np.transpose(
            rolling_window(
                np.transpose(array, axes=(1, 0, 2)), 2),
            axes=(0, 1, 3, 2)
        )

        # indices for all possible pairs
        i_triu = np.transpose(np.triu_indices(d.shape[1], 1))

        r = d[:, i_triu, :]
        v = r[:, :, :, 1] - r[:, :, :, 0]
        p = r[:, :, 1, 0] - r[:, :, 0, 0]

        # calculates lambda
        l = (v[:, :, 1, 0] * p[:, :, 1] - v[:, :, 1, 1] * p[:, :, 0]) / (
                v[:, :, 0, 1] * v[:, :, 1, 0] - v[:, :, 0, 0] * v[:, :, 1, 1])

        I = (l < 1) & (l > 0)  # | (np.arange(d.shape[0])[:, None] >= d.shape[0]-2)
        l[~I] = np.nan

        k = np.ones_like(l)
        k[~I] = np.nan

        crossings = l[:, :, None] * v[:, :, 0, :] + r[:, :, 0, 0] * k[:, :, None]
        crossings = np.transpose(crossings, (1, 0, 2))

        return crossings, properties_array[i_triu[:, 0], :], properties_array[i_triu[:, 1], :]

    def plot(self, ax: Axes, **kwargs):

        props = plotting.ray_properties.copy()
        props.update(kwargs)

        self.wavelength[np.isnan(self.wavelength)] = 0.
        self.group[np.isnan(self.group)] = 0.

        plt_groups = np.unique(self.properties_array, axis=0)

        if len(plt_groups) > 1:
            lines = list()
            if (plt_groups[:, 1] == 0.).all():
                prop_cycle = iter(rcParams['axes.prop_cycle'])
                for plt_props in plt_groups:
                    group_props = props.copy()
                    group_props.update({'color': next(prop_cycle)['color']})
                    i = (self.properties_array == plt_props).all(axis=1)
                    lines += ax.plot(self.x[i, :].T, self.y[i, :].T, **group_props)

            else:
                prop_cycle = iter(cycle(['-', '--', '-.', ':']))
                g_map = {g: next(prop_cycle) for g in np.unique(plt_groups[:, 0])}
                linestyles = list(map(lambda g: g_map[g], plt_groups[:, 0]))
                for i, plt_props in enumerate(plt_groups):
                    _, w = plt_props
                    group_props = props.copy()
                    if w != 0:
                        group_props.update({'color': wavelength_to_rgb(w)})
                    group_props.update({'linestyle': linestyles[i]})
                    i = (self.properties_array == plt_props).all(axis=1)
                    lines += ax.plot(self.x[i, :].T, self.y[i, :].T, **group_props)

        elif len(plt_groups) > 0:
            w = plt_groups[0][1]
            group_props = props.copy()
            if w != 0:
                group_props.update({'color': wavelength_to_rgb(w)})
            lines = ax.plot(self.x.T, self.y.T, **group_props)
        else:
            lines = ax.plot(self.x.T, self.y.T, **props)

        return lines


class RayCrossings1D:

    @property
    def n(self):
        return self.array.shape[0]

    def __init__(self, array: np.array, properties_from: np.array, properties_to: np.array):
        self.array, self.properties_from, self.properties_to = array, properties_from, properties_to

    def __getitem__(self, ix):
        return RayCrossings1D(self.array[ix, :],
                              self.properties_from[ix, :],
                              self.properties_to[ix, :])

    x = _view_property(slice(None), 0)
    y = _view_property(slice(None), 1)
    points = _view_property(slice(None), slice(0, 2))

    group_from = _view_property(slice(None), 0, attr='properties_from')
    wavelength_from = _view_property(slice(None), 1, attr='properties_from')

    group_to = _view_property(slice(None), 0, attr='properties_to')
    wavelength_to = _view_property(slice(None), 1, attr='properties_to')

    def image_crossings(self):
        i = np.zeros_like(self.wavelength_to).astype(bool)
        for g, w in np.unique(self.properties_from, axis=0):
            i |= ((self.wavelength_from == w) & (self.wavelength_to == w) & (self.group_from == g) & (
                    self.group_to == g))
        return self[i]

    def color_crossings(self):
        i = np.zeros_like(self.wavelength_to).astype(bool)
        for g, w in np.unique(self.properties_from, axis=0):
            i |= ((self.wavelength_from == w) & (self.wavelength_to == w) & (self.group_from == g) & (
                    self.group_to != g))
        return self[i]


class RayCrossings(RayCrossings1D):

    @staticmethod
    def from_traced_rays(traced_rays: TracedRays, element=None):
        return RayCrossings(*traced_rays.ray_crossings(element))

    def before(self, element: int):
        return RayCrossings1D(self.array[:, element, :], self.properties_from, self.properties_to)

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            ix, iy = item
            if isinstance(iy, slice):
                return RayCrossings(self.array[ix, iy, :],
                                    self.properties_from[ix, :],
                                    self.properties_to[ix, :])
            elif isinstance(iy, int):
                return RayCrossings1D(self.array[ix, iy, :],
                                      self.properties_from[ix, :],
                                      self.properties_to[ix, :])
            else:
                raise IndexError()
        else:
            return self.__getitem__((item, slice(None)))

    x = _view_property(slice(None), slice(None), 0)
    y = _view_property(slice(None), slice(None), 1)
    points = _view_property(slice(None), slice(None), slice(0, 2))


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
    dx[np.isnan(dx)] = 0.
    rays.y[np.logical_xor(dx > 0, rays.forward > 0)] = np.nan

    return rays


def point_source_rays(origin=(0., 0.), angle=(-50., 50.), n: int = 9, group: int = None):
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

    da = (max(angle) - min(angle)) / float(n - 1)
    rays = Rays(np.zeros((n, 4)))
    rays.points = origin[None, :]
    angle = np.arange(0, n) * da + min(angle)
    rays.tan_theta = np.tan(angle * np.pi / 180.)
    m = np.floor(angle / 360.)
    angle = (angle - m * 360.)
    rays.forward = ((angle < 90.) | (angle > 270.)).astype(float)

    if group is None:
        group = uuid.uuid1().int >> 64

    rays.group = np.array(group).view(dtype=np.float)

    return rays
