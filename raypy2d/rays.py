import numpy as np


def propagate(rays, x: float, stop : int = 1):
    """
    Propagates the rays in free space up to the x
    Args:
        rays: (np.array) rays to propagate
        x: (float) x-coordinate of the plane up to which the rays should propagate
        stop: (int) either 1 or -1

    Returns:
        (np.array) propagated rays

    """

    dx = x - rays[:, 0]

    rays[:, 1] = rays[:, 1] + rays[:, 2] * dx
    #rays[(stop * dx)<0, 2] = np.nan
    rays[:, 0] = x

    return rays


def ray_fan(origin=[0., 0.], angle=[-90., 90.], n=9):

    origin = np.array(origin)

    da = (angle[1]-angle[0]) / float(n+1)
    rays = np.zeros((n, 3))
    rays[:,:2] = origin[None, :]
    rays[:, 2] = np.tan((np.arange(1, n+1)*da + angle[0]) * np.pi/180.)

    return rays
