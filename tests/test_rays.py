import numpy as np
from raypy2d.rays import Rays, TracedRays
from raypy2d.paths import OpticalPath
import pytest


@pytest.fixture
def demo_path():
    from raypy2d.elements import Aperture, ParabolicMirror, DiffractionGrating, Lens, Sensor
    from raypy2d.paths import Object, OpticalPath

    path = OpticalPath(Object(2.0, [-8., 0.], angle=[-20, 20], n=181))

    # path.append(Aperture(1, [6.0, 0], blocker_diameter=20))
    path.append(Aperture(0.2, [0.0, 0], blocker_diameter=20))

    path.append(ParabolicMirror(40, 20., [32., 0], theta=155, flipped=True))
    # path.append(Mirror(20., [50., 0], theta=165, flipped=True))
    # path.append(DiffractionGrating(1.6, 20., interference=-1, theta=-10), distance=15., theta=130.)
    path.append(DiffractionGrating(1.0, 20., interference=1, theta=-10), distance=20., theta=133)
    # path.append(Mirror(20., flipped=True, theta=30), distance=20., theta=105)
    alpha = -5
    vec = np.array([np.cos(alpha / 180. * np.pi), np.sin(alpha / 180. * np.pi)])
    sin_alpha = np.cos(alpha / 180. * np.pi)
    path.append(Aperture(13.75, theta=alpha, blocker_diameter=28),
                Lens(28.0, 13.75, vec * 4, theta=alpha, flipped=False),
                # Lens(12.0, 11, vec*20, theta=alpha, flipped=False),
                Sensor(5.58, 30 * vec, theta=alpha, flipped=True), distance=13., theta=85)

    return path


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

    tr = TracedRays.from_rays(rays)

    assert tr.array.shape == (10, 2, 3)
    assert tr.x.shape == (10, 2)


def test_traced_rays_index(demo_path: OpticalPath):
    tr = demo_path.rays.traced_rays()
    assert tr.n == 1629
    tr = tr[::2, 5:6]
    assert tr.n == 815
    assert tr.points.shape == (815, 1, 2)
    assert tr.wavelength.shape == (815,)


def test_ray_crossings_index(demo_path: OpticalPath):
    tr = demo_path.rays.ray_crossings(5)
    assert tr.array.shape == (1225, 7, 2)
    assert tr.n == 1225
    tr = tr[::2, 5:6]
    assert tr.n == 613
    assert tr.properties_to.shape == (613, 2)
    assert tr.properties_from.shape == (613, 2)
    assert tr.points.shape == (613, 1, 2)
    assert tr.wavelength_to.shape == (613,)


def test_ray_crossings(demo_path):
    r = demo_path.rays.ray_crossings(5)
    assert np.unique(r.properties_from, axis=0).shape[0] > 1

    im = r.image_crossings().before(5)

    assert im.n == 147
