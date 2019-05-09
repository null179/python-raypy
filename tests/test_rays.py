import numpy as np
from raypy2d.rays import Rays


def test_rays_object():
    """
    test the ray interpretation of a numpy array
    """

    arr = np.random.rand(5,5)

    rays = Rays(arr)

    assert rays.array is arr
    rays.points[:,0] = 0.
    assert (arr[:,0] == 0.).all()
    assert (arr[:,1] == rays.points[:,1]).all()

    rays.y = 3
    assert (arr[:,1] == 3.).all()

    assert (rays.za.shape[1] == 2)
    assert (rays.za == arr[:,1:3]).all()
