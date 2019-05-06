import numpy as np
from .elements import Element
from .utils import rotation_matrix
from .rays import ray_fan, propagate


class ImagePath:

    def __init__(self):

        self.elements = []
        self.rays = [ray_fan([0., 1.], angle=[-50, 50])]

    def append(self, element: Element):
        self.elements.append(element)
        self.rays.append(element.trace(self.rays[-1].copy()))

    def propagate(self, x):
        self.rays.append(propagate(self.rays[-1].copy(), x))

    def plot(self, ax):

        # plot rays
        rays = np.array(self.rays)
        ax.plot(rays[:, :, 0], rays[:, :, 1], color='orange')

        # plot all elements
        for element in self.elements:
            element.plot(ax)


class Object:

    def __init__(self, height, origin=[0., 0.], theta: float = 0.):
        """
        Creates an object subject to imaging. The object emits three fans of rays
        Args:
            height: (float) height of the object
            origin: position of the object
            theta: (float) rotation angle in degrees
        """

        self.height = height
        self.origin = np.array(origin)
        self.theta = theta
