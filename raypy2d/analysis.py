import numpy as np
from matplotlib.axes import Axes

from raypy2d.paths import OpticalPath
from raypy2d.utils import wavelength_to_rgb


def plot_sensor_img(path: OpticalPath, ax: Axes, only_wavelength=False, pixel=3280):
    """
    Plots a ray path and plots the ray distribution on the last element in the path
    Args:
        path: (OpticalPath)
        ax: (matplotlib.Axes) axes to plot the path in
        only_wavelength: (bool)
        pixel: (int) number of pixels the sensor have

    Returns:
        (pandas.Dataframe) with sensor image
    """
    try:
        import pandas as pd
    except ImportError:
        ax.text(0, 0, "This function needs pandas library!")

    # assumes the last element in the path to be the sensor element
    s = path.elements[-1]

    sensor_image = s.to_element_frame_of_reference(path.rays)

    mm_px = s.diameter / pixel

    img = pd.DataFrame(
        sensor_image.array,
        columns=['x', 'y', 'tan_theta', 'forward', 'group', 'wavelength'])

    img = img.dropna(subset=['x', 'y', 'tan_theta']).fillna(0)
    efficiency = img.shape[0] / sensor_image.array.shape[0]

    if not only_wavelength:
        img = img.groupby(['group', 'wavelength'])
    else:
        img = img.groupby('wavelength')

    img = img.y.agg(['mean', 'std', 'min', 'max'])
    img = img.reset_index()

    x = np.linspace(-s.diameter / 2., s.diameter / 2., pixel)
    y = np.exp(-0.5 * ((x[:, None] - img['mean'].values[None, :]) / img['std'].values[None, :]) ** 2)

    g = np.zeros_like(y)

    for i in range(g.shape[1]):
        g[np.argmin((x[:, None] - img['mean'].values[None, :]) ** 2, axis=0)[i], i] = 1.5

    size = (img['mean'].max() - img['mean'].min()) / s.diameter

    ax.set_ylim(-0.1, 2.5)
    ax.set_xlim(-s.diameter / 2., s.diameter / 2.)
    ax.text(-s.diameter / 2. * 0.9, 2, "size: {:.1f}% of sensor (efficiency {:.1f}%)".format(size * 100,
                                                                                             efficiency * 100))
    for i in range(g.shape[1]):
        if img.wavelength.values[i] != 0.:
            c = wavelength_to_rgb(img.wavelength.values[i])
        else:
            c = None
        line, = ax.plot(x, y[:, i], color=c)
        ax.plot(x, g[:, i], color=line.get_color())
        ax.text(img['mean'].values[i], 1.6, "{:.1f}px".format((img['max'] - img['min']).values[i] / mm_px),
                ha=' center')

    return sensor_image
