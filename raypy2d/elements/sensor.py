import numpy as np
from .aperture import Aperture
from .mirror import Mirror


class Sensor(Mirror):

    def __init__(self, diameter: float, origin=[0., 0.], theta=0., blocker_diameter: float = float('+Inf'),
                 flipped=False):
        """
        Creates a sensor element
        Args:
            diameter: (float) diameter of the lens
            origin: position of the center of the lens
            theta: rotation angle of mirror (with respect the abscissa)
            blocker_diameter: (float, optional) size of the aperture blocker
            flipped: (bool) if the edges should be flipped or not
        """

        Aperture.__init__(self, diameter, origin, theta, blocker_diameter, flipped)
        self.matrix = np.diag([1., 1.])
        self.mirroring = False