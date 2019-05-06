from matplotlib.patches import Arrow
from matplotlib import rc_params

import numpy as np
from .elements import Element, RotateObject
from .rays import ray_fan, propagate
from .utils import wavelength_to_rgb


class Object(RotateObject):

    def __init__(self, height, origin=[0., 0.], theta: float = 0., fans=[0, 0.5, 1.0], n_rays: int = 9):
        """
        Creates an object subject to imaging. The object emits three fans of rays
        Args:
            height: (float) height of the object
            origin: position of the object
            theta: (float) rotation angle in degrees
            fans: (list[float]) position of the ray fans emitted from object
            n_rays: (int) number of rays per fan
        """

        RotateObject.__init__(self, origin, theta)
        self.height = height
        self.fans_at = fans

        self.rays = []
        for i, fan in enumerate(self.fans_at):

            y0 = fan * self.height - self.height / 2.

            rays = ray_fan([0, y0], [-75, 75], n=n_rays)
            rays = np.append(rays, np.ones((rays.shape[0], 1)) * i, axis=1)
            rays = self.to_global_frame_of_reference(rays)
            self.rays.append(rays)

        self.rays = np.vstack(self.rays)

        # transform
        self.rays = self.to_global_frame_of_reference(self.rays)

    def plot(self, ax):

        points = np.array([[0,  self.height],
                           [0, -self.height]]) / 2.

        points = self.points_to_global_frame_of_reference(points)

        arrow = Arrow(points[0, 0], points[0, 1],
                      dx=points[1, 0]-points[0, 0],
                      dy=points[1, 1]-points[0, 1],
                      color='blue')

        ax.add_patch(arrow)


class ImagePath:

    def __init__(self, obj: Object=None):

        self.elements = []
        self.obj = obj
        if self.obj is None:
            self.rays = [ray_fan([0., 0.], angle=[-50, 50])]
        else:
            self.rays = [obj.rays]

    def append(self, element: Element):
        self.elements.append(element)
        self.rays.append(element.trace(self.rays[-1].copy()))

    def propagate(self, x):
        self.rays.append(propagate(self.rays[-1].copy(), x))

    def _3d_array_of_rays(self):

        cols = max([arr.shape[1] for arr in self.rays])
        rows = max([arr.shape[0] for arr in self.rays])

        arrs = []
        for rayarr in self.rays:
            new_cols= cols-rayarr.shape[1]
            new_rows= rows-rayarr.shape[0]

            if new_cols > 0:
                rayarr = np.hstack((rayarr, np.ones((rayarr.shape[0], new_cols))))
                rayarr[:,-new_cols:] = np.nan

            if new_rows > 0:
                rayarr = np.vstack((rayarr, np.ones((new_rows, rayarr.shape[1]))))
                rayarr[-new_rows:, :] = np.nan

            arrs.append(rayarr)

        return np.asarray(arrs)

    def plot(self, ax):

        # plot rays
        rays = self._3d_array_of_rays()

        if 4 < rays.shape[2]:
            for i in range(rays.shape[1]):
                color = rays[:, i, 4]
                color = color[~np.isnan(color)][0]
                color = wavelength_to_rgb(color)
                linestyle = rays[:, i, 3]
                linestyle = linestyle[~np.isnan(linestyle)].astype(int)[0]
                linestyle = (['--', '-','-.']*10)[linestyle]
                ax.plot(rays[:, i, 0], rays[:, i, 1], color = color, linestyle = linestyle)

        elif 3 < rays.shape[2]:

            cycler = iter(rc_params()['axes.prop_cycle'])

            for c in set(rays[0, : , 3].tolist()):
                I = (rays[0, :, 3]==c).squeeze()
                ax.plot(rays[:, I, 0], rays[:, I, 1], color = next(cycler)['color'])
        else:
            ax.plot(rays[:, :, 0], rays[:, :, 1], color='orange')

        # plot all elements
        for element in self.elements:
            element.plot(ax)

        # plot object
        if self.obj is not None:
            self.obj.plot(ax)