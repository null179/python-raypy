from matplotlib import pyplot as plt
import raypy2d
from raypy2d import plotting
from raypy2d.elements import Aperture, Lens, ParabolicMirror, Mirror, DiffractionGrating, Sensor
from raypy2d.rays import propagate, point_source_rays
from raypy2d.paths import OpticalPath, Object
import numpy as np


def test_group_elements():

    path = OpticalPath(angle=[-5, 5], n=31)
    raypy2d.elements.plot_blockers = False

    path.append(Aperture(0.1, [8.0, 0], blocker_diameter=20))

    path.append(ParabolicMirror(32, 12., [40., 0], theta=175, flipped=True))
    path.append(DiffractionGrating(1.6, 10., interference=-1, theta=-10), distance=20., theta=170.)
    path.append(Mirror(15., theta=205.8, flipped=False), distance=12, theta=129)

    path.append(Aperture(13.75, flipped=True, blocker_diameter=15),
                Lens(22.0, 13.75, [0.01, 0], flipped=False),
                Lens(6.2, 13.75, [0.02, 0], flipped=False),
                Mirror(3.68, [3.80, 0], flipped=True), distance=30.)

    ax = plt.gca()
    ax.axis('equal')
    path.plot(ax)
    plt.show()


def test_sensor_element():

    path = OpticalPath(angle=[-5, 5], n=31)
    raypy2d.elements.plot_blockers = False

    path.append(Aperture(0.1, [8.0, 0], blocker_diameter=20))

    path.append(ParabolicMirror(32, 12., [40., 0], theta=175, flipped=True))
    path.append(DiffractionGrating(1.6, 10., interference=-1, theta=-10), distance=20., theta=170.)
    path.append(Mirror(15., theta=205.8, flipped=False), distance=12, theta=129)

    path.append(Aperture(13.75, flipped=True, blocker_diameter=15),
                Lens(22.0, 13.75, [0.01, 0], flipped=False),
                Lens(6.2, 13.75, [0.02, 0], flipped=False),
                Sensor(3.68, [3.80, 0], flipped=True), distance=30.)

    ax = plt.gca()
    ax.axis('equal')
    path.plot(ax)

    cross = path.rays.traced_rays().ray_crossings()
    cross = cross.reshape((-1, 2))
    cross = cross[~np.any(np.isnan(cross), axis=1)]

    ax.scatter(cross[:, 0], cross[:, 1])

    plt.show()