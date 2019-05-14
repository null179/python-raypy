import numpy as np
from raypy2d.rays import Rays, TracedRays


def test_rays_object():
    """
    test the ray interpretation of a numpy array
    """

    arr = np.random.rand(5, 6)

    rays = Rays(arr)

    assert rays.array is arr
    assert rays.arrays[-1] is arr
    rays.points[:, 0] = 0.
    assert (arr[:, 0] == 0.).all()
    assert (arr[:, 1] == rays.points[:, 1]).all()

    rays.y = 3
    assert (arr[:, 1] == 3.).all()

    assert (rays.za.shape[1] == 2)
    assert (rays.za == arr[:, 1:3]).all()

    rays.store()

    rays.x += 1

    assert arr is rays.array
    assert arr is rays.arrays[-1]
    assert arr is not rays.arrays[-2]
    assert (arr[:, 1:] == rays.array[:, 1:]).all()


def test_traced_rays_object():
    arr = np.random.rand(10, 3)

    rays = Rays(arr)

    assert rays.array.shape == (10, 6)

    rays.store()
    rays.x = np.random.rand(10)
    rays.group = 3

    tr = TracedRays(rays)

    assert tr.array.shape == (10, 2, 3)
    assert tr.x.shape == (10, 2)
