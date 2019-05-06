import numpy as np
from .utils import rotation_matrix


class ImagePath:

    pass

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
