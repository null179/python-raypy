import numpy as np

def _view_property(x_slice: slice, y_slice: slice):
    def _get(self):
        return self.array[x_slice, y_slice]

    def _set_x(self, x):
        self.array[x_slice, y_slice] = x

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
        assert array.shape[1] >= 3

        # store a view to array
        self.array = array

    x = _view_property(slice(None), 0)
    y = _view_property(slice(None), 1)
    tan_theta = _view_property(slice(None), 2)
    group = _view_property(slice(None), 3)
    wavelength = _view_property(slice(None), 4)
    points = _view_property(slice(None), slice(None, 2))
    za = _view_property(slice(None), slice(1,3))



def propagate(rays: Rays, x: float):
    """
    Propagates the rays in free space up to the x
    Args:
        rays: (Rays) rays to propagate
        x: (float) x-coordinate of the plane up to which the rays should propagate

    Returns:
        (np.array) propagated rays

    """

    dx = x - rays.x

    rays.y = rays.x + rays.tan_theta * dx
    rays.x = x

    return rays


def ray_fan(origin=[0., 0.], angle=[-90., 90.], n=9):

    origin = np.array(origin)

    da = (angle[1]-angle[0]) / float(n+1)
    rays = Rays(np.zeros((n, 3)))
    rays.points = origin[None, :]
    rays.tan_theta = np.tan((np.arange(1, n+1)*da + angle[0]) * np.pi/180.)

    return rays
