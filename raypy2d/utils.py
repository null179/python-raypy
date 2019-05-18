import numpy as np


def rotation_matrix(theta: float):
    """
    Returns the rotation 2D matrix for the passed angle
    Args:
        theta: (float) angle in degrees of the rotation matrix

    Returns:
        (numpy.array) 2x2 rotation matrix

    """

    cos_theta = np.cos(theta * np.pi / 180.)
    sin_theta = np.sin(theta * np.pi / 180.)
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


def wavelength_to_rgb(wavelength, gamma=0.8):
    """
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    """

    wavelength = float(wavelength)
    if 380 <= wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 <= wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return R, G, B


def assure_number_of_columns(array: np.array, n_columns: int):
    """
    Add columns with NaN if the passed array has less then the passed number of columns
    Args:
        array: (numpy.array) two-dimensional array
        n_columns: (int) minimal number of columns

    Returns:
        (numpy.array) two-dimensional array with minimal number of columns

    """

    new_cols = n_columns - array.shape[1]

    if new_cols > 0:
        array = np.hstack((array, np.zeros((array.shape[0], new_cols))))
        array[:, -new_cols:] = np.nan

    return array


def assure_number_of_rows(array: np.array, n_rows: int):
    """
    Add rows with NaN if the passed array has less then the passed number of columns
    Args:
        array: (numpy.array) two-dimensional array
        n_rows: (int) minimal number of rows

    Returns:
        (numpy.array) two-dimensional array with minimal number of rows

    """

    new_rows = n_rows - array.shape[0]

    if new_rows > 0:
        array = np.vstack((array, np.zeros((new_rows, array.shape[1]))))
        array[-new_rows:, :] = np.nan

    return array


def rolling_window(a, window):
    """
    creates a view of an array for rolling windows
    Args:
        a: (numpy.array) array dimension d of shape (n,...)
        window: (int) length w of the window

    Returns:
        (numpy.array) view with shape (n-w+1, ..., w)
    """

    # transpose first dimension to last
    axes = tuple(range(1, len(a.shape))) + (0,)
    invaxes = (-2,) + tuple(range(0,len(a.shape)-1)) + (-1,)

    a = np.transpose(a, axes)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    view = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    view = np.transpose(view, invaxes)
    return view


def place_relative_to(reference_element, element, distance, theta):
    """
    Place element relative to the reference element with a distance and an angle
    Args:
        reference_element: (Element) reference element
        element: (Element) element to be place
        distance: (float) distance to reference element
        theta:  (float) angle with respect to reference element
    """
    offset = reference_element.origin
    diff = np.array([distance, 0.])

    if theta != 0.:
        Rmat = rotation_matrix(-theta)
        diff = np.dot(diff, Rmat).squeeze()
        element.origin = np.dot(element.origin[None,:], Rmat).squeeze()

    offset += diff

    element.origin += offset

    element.theta += theta
