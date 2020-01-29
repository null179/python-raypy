from raypy2d.elements import DiffractionGrating, DiffractionPrism, Aperture, ParabolicMirror, Lens, Sensor
from raypy2d.paths import OpticalPath, Object
from matplotlib import pyplot as plt
import numpy as np
import pytest


@pytest.fixture
def path_with_parabolic_mirrors():
    NA = 0.22
    alpha = np.arcsin(NA) / np.pi * 180.

    path = OpticalPath(Object(0.3, [0., 0.], angle=[alpha, -alpha], n=31))

    pm = ParabolicMirror(30, 20, theta=-45 + 180, flipped=False)
    pm.mirroring = False
    path.append(pm, distance=20)
    # path.append(Mirror(10, theta=-45, flipped=True), distance=30)
    theta_0 = 2 * -path.elements[-1].theta
    path.elements

    # path.append(Aperture(1, [6.0, 0], blocker_diameter=20))

    pm = ParabolicMirror(30, 20., theta=152, flipped=True)
    path.append(pm, distance=30, theta=theta_0)
    # path.append(Mirror(20., [50., 0], theta=165, flipped=True))
    # path.append(DiffractionGrating(1.6, 20., interference=-1, theta=-10), distance=15., theta=130.)

    theta_0 = 2 * path.elements[-1].theta - (180 + theta_0)
    theta_diff = 5
    m_diff = 1

    path.append(DiffractionGrating(1.0, 20., interference=m_diff, theta=theta_diff), distance=17., theta=theta_0)

    theta_532 = path.elements[-1].diffraction_angle_for(532., theta_diff)

    # path.append(Mirror(20., flipped=True, theta=30), distance=20., theta=105)
    alpha = 0
    vec = np.array([np.cos(alpha / 180. * np.pi), np.sin(alpha / 180. * np.pi)])
    path.append(Aperture(13.75, theta=alpha, blocker_diameter=28),
                Lens(15.0, 13.75, vec * 4, theta=alpha, flipped=False),
                # Lens(12.0, 11, vec*20, theta=alpha, flipped=False),
                Sensor(5.58, 19.68 * vec, theta=alpha, flipped=True), distance=10., theta=theta_0 + theta_532)

    return path


def test_parabolic_mirror(path_with_parabolic_mirrors):
    """
    test the diffraction of rays on a grating
    """
    path = path_with_parabolic_mirrors

    ax = plt.gca()
    ax.axis('equal')
    path.plot(ax)
    plt.show()


def test_diffraction_grating():
    """
    test the diffraction of rays on a grating
    """
    path = OpticalPath(Object(1, theta=0))

    theta_diff = -30
    m_diff = 1

    path.append(DiffractionGrating(1.0, 15, origin=[10., 0], theta=theta_diff))

    theta_532 = np.arcsin(np.sin(-theta_diff / 180. * np.pi)
                          - m_diff * 532. / 1000. / path.elements[-1].grating) * 180 / np.pi \
                + theta_diff

    theta_532_2 = path.elements[-1].diffraction_angle_for(532., theta_diff)

    assert np.isclose(theta_532, -31.8337)
    assert np.isclose(theta_532_2, -31.8337)

    # path.append(Aperture(20.), distance=20., theta=theta_532)
    # ax = plt.gca()
    # ax.axis('equal')
    # path.plot(ax)
    # plt.show()


def test_diffraction_prism():
    """
    test the diffraction of rays on a prism
    """
    path = OpticalPath(Object(1, theta=0))

    theta_diff = -30
    m_diff = 1

    path.append(DiffractionPrism(15, origin=[10., 0], theta=theta_diff))

    theta_532 = np.arcsin(np.sin(-theta_diff / 180. * np.pi)
                          - m_diff * 532. / 1000. / 1.0) * 180 / np.pi \
                + theta_diff

    assert np.isclose(theta_532, -31.8337)

    path.append(Aperture(20.), distance=20., theta=theta_532)
    ax = plt.gca()
    ax.axis('equal')
    path.plot(ax)
    plt.show()
