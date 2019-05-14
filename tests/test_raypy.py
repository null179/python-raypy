from matplotlib import pyplot as plt
import raypy2d
from raypy2d import plotting
from raypy2d.elements import Aperture, Lens, ParabolicMirror, Mirror, DiffractionGrating
from raypy2d.rays import propagate, point_source_rays
from raypy2d.paths import OpticalPath, Object
import numpy as np


def test_plot_aperture():
    """
    test the plot function of the aperture
    """
    ax = plt.gca()
    ax.axis('equal')
    a = Aperture(diameter=5., origin=[1., 3.], theta=np.pi / 2.34, blocker_diameter=10.)
    a.plot(ax)
    plt.show()


def test_plot_lens():
    """
    test the plot function of the lens
    """
    ax = plt.gca()
    ax.axis('equal')
    a = Lens(focal_length=10., diameter=5., origin=[1., 3.], theta=20.34, blocker_diameter=10.)
    a.plot(ax)
    plt.show()


def test_some_rays():
    first_element = Aperture(5)
    second_element = Lens(5, 16., [5., 1.], theta=30., blocker_diameter=18.)
    third_element = Lens(10, 8., [10., 3.], theta=-15, blocker_diameter=10.)

    rays = []

    r = point_source_rays([-4., 1.], angle=[-50., 50], n=21)

    # r = np.array([[0., 1., 45.*np.pi/180.]])
    # r = first_element.to_global_frame_of_reference(r)
    rays.append(r.array.copy())
    r = first_element.trace(r)
    rays.append(r.array.copy())
    r = second_element.trace(r)
    rays.append(r.array.copy())
    r = third_element.trace(r)
    rays.append(r.array.copy())
    r = propagate(r, 35)
    rays.append(r.array.copy())
    rays = np.array(rays)

    ax = plt.gca()
    ax.axis('equal')
    ax.plot(rays[:, :, 0], rays[:, :, 1], color='red', linewidth=0.5)
    first_element.plot(ax)
    second_element.plot(ax)
    third_element.plot(ax)
    plt.show()


def test_some_rays_with_mirror():
    first_element = Lens(5, 16., [5., 1.], theta=30.)
    second_element = Aperture(4, [7., 2.])
    third_element = Mirror(8., [15., 3.], theta=160)
    fourth_element = ParabolicMirror(5., 8., [9., 10.])

    r = point_source_rays([0., 1.], angle=[-50, 50])

    r.store()
    r = first_element.trace(r)
    r.store()
    r = second_element.trace(r)
    r.store()
    r = third_element.trace(r)
    r.store()
    r = fourth_element.trace(r)
    r.store()
    r = propagate(r, 35)
    rays = r.traced_rays()

    raypy2d.elements.plot_blockers = False

    ax = plt.gca()
    ax.axis('equal')
    rays.plot(ax, color='orange')
    first_element.plot(ax)
    second_element.plot(ax)
    third_element.plot(ax)
    fourth_element.plot(ax)
    plt.show()


def test_imaging_path():
    path = OpticalPath()

    path.append(Lens(5, 16., [5., 1.], theta=30.))
    path.append(Aperture(4, [7., 2.]))
    path.append(ParabolicMirror(5., 8., [15., 3.], theta=155))
    path.append(ParabolicMirror(5., 8., [9., 15.], theta=-45))
    path.propagate(15)

    ax = plt.gca()
    ax.axis('equal')
    path.plot(ax)
    plt.show()


def test_imaging_path_with_object():
    obj = Object(2.0, theta=15.)
    path = OpticalPath(obj)

    path.append(Lens(3, 16., [5., 1.], theta=30.))
    path.append(Lens(10, 16., [8., 3.], theta=15.))
    path.propagate(20)

    ax = plt.gca()
    ax.axis('equal')
    path.plot(ax)
    plt.show()


def test_diffraction_grating():
    path = OpticalPath()

    path.append(Lens(3, 16., [5., 1.], theta=30.))
    path.append(Lens(10, 16., [8., 3.], theta=15.))
    path.append(DiffractionGrating(1.6, 16., [10, 3.], theta=15.))
    path.propagate(25)

    ax = plt.gca()
    ax.axis('equal')
    path.plot(ax)
    plt.show()


def test_imaging_path_with_diffraction_grating():

    raypy2d.elements.plot_blockers = False

    obj = Object(2.0, n_rays=5)
    path = OpticalPath(obj)

    path.append(Lens(3, 16., [3., 0]))
    path.append(DiffractionGrating(1.6, 16., [8, 0.]))
    # path.append(Lens(5, 16., [10.,0]))
    path.append(ParabolicMirror(16., 35., [20, 5.5], theta=160., flipped=True))

    # path.append(Lens(3, 16., [15, 1.], theta=10.))
    path.propagate(-5)

    ax = plt.gca()
    ax.axis('equal')
    path.plot(ax)
    plt.show()


def test_max_aperture():
    first_element = Aperture(5)
    second_element = Lens(5, 16., [5., 1.], theta=30., blocker_diameter=18.)
    third_element = Lens(10, 8., [10., 3.], theta=-15, blocker_diameter=10.)

    raypy2d.elements.plot_blockers = False

    ax = plt.gca()
    ax.axis('equal')
    first_element.plot(ax)
    second_element.plot(ax)
    third_element.plot(ax)
    plotting.plot_maximal_aperture(ax, first_element, second_element)
    plotting.plot_maximal_aperture(ax, second_element, third_element)
    plt.show()
