import numpy as np
from .helper import convertTo


__all__ = ["getLuminance"]


def getLuminance(img_data, bgrWeight):
    '''
    Get the luminance of input image with given BGR weight. Computation detail:

    bgrWeight is normalized
    img_data is converted to np.float64
    y = np.einsum("i,...i->...", bgrWeight, img_data)
    l[y > 0.008856] = 116.0 * y ** (1/3) - 16.0
    l[y < 0.008856] = 903.3 * y
    lum = l * 0.01
    lum is converted back to img_data.dtype and returned

    Parameters
    ----------
    img_data: 3D numpy.ndarray
        The input image data

    bgrWeight: numpy.ndarray with shape == (3,)
        Weight for the three channels

    Returns
    -------
    lum: 2D numpy.ndarray
        Lumiance of the image
    '''
    w = bgrWeight / np.sum(bgrWeight)
    tmp = convertTo(img_data, np.float64)
    tmp = np.einsum("i,...i->...", w, tmp)
    lum = np.empty_like(tmp)
    lum[tmp > 0.008856] = 116.0 * (tmp[tmp > 0.008856] ** (1.0 / 3.0)) - 16.0
    lum[tmp <= 0.008856] = 903.3 * tmp[tmp <= 0.008856]
    lum = convertTo(lum * 0.01, img_data.dtype)
    return lum
