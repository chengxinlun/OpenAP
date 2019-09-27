import numpy as np
from .helper import convertTo


__all__ = ["chromaticAdaptation", "colorConvert", "getLuminance", "getY",
           "linearizeRGB"]


# White point
white_D50 = np.array([0.96422, 1.0, 0.82521], dtype=np.float64)
white_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
white_E = np.array([1.0, 1.0, 1.0], dtype=np.float64)
# Response matrix
response_XYZScaling = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float64)
response_Bradford = np.array([[0.8951, 0.2664, -0.1614],
                              [-0.7502, 1.7135, 0.0367],
                              [0.0389, -0.0685, 1.0296]], dtype=np.float64)
response_VonKries = np.array([[0.40024, 0.7076, -0.08081],
                              [-0.2263, 1.16532, 0.0457],
                              [0.0, 0.0, 0.91822]], dtype=np.float64)
# Color convert
convert_sRGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                             [0.2126729, 0.7151522, 0.0721750],
                             [0.0193339, 0.1191920, 0.9503041]],
                            dtype=np.float64)


def linearizeRGB(img_data):
    '''
    Linearize RGB from sRGB to linear RGB in D65. Computation detail:

        if u <= 0.04045:
            u / 12.92
        else:
            ((u + 0.055) / 1.055) ** 2.4

    Parameters
    ----------
    img_data: 2D or 3D numpy.array
        Input image data

    Returns
    -------
    Linear RGB image with D65 white point since sRGB is in D65
    '''
    return np.piecewise(img_data, [img_data <= 0.04045, img_data > 0.04045],
                        [lambda x: x / 12.92,
                         lambda x: ((x + 0.055) / 1.055) ** 2.4])


def chromaticAdaptation(img_data, sourceWhite, destinationWhite, response):
    '''
    Chromatic adaptation

    From xyz in one white point to xyz in another white point

    Reference: http://brucelindbloom.com/index.html?Eqn_ChromAdapt.html
    '''
    rho_source = np.einsum("ij, j->i", response, sourceWhite)
    rho_destin = np.einsum("ij, j->i", response, destinationWhite)
    rho_ratio = rho_destin / rho_source
    rho_ratio = np.diag(rho_ratio)
    color_matrix = np.linalg.inv(response) @ rho_ratio @ response
    img_data_adapted = np.einsum("...i, ji->...j", img_data, color_matrix)
    # Clip the output due to float precision
    np.clip(img_data_adapted, 0.0, 1.0, out=img_data_adapted)
    return img_data_adapted


def colorConvert(img_data, convertMatrix):
    tmp = np.einsum("...i, ji->...j", img_data, convertMatrix)
    # Clip the output due to float precision
    np.clip(tmp, 0.0, 1.0, out=tmp)
    return tmp


def getLuminance(img_data):
    '''
    Get the luminance of input image. Input image must be in XYZ colorspace.
    Computation detail:

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
    tmp = img_data[:, :, 1]
    lum = np.empty_like(tmp)
    lum[tmp > 0.008856] = 116.0 * (tmp[tmp > 0.008856] ** (1.0 / 3.0)) - 16.0
    lum[tmp <= 0.008856] = 903.3 * tmp[tmp <= 0.008856]
    lum = lum * 0.01
    # Clip the output due to float precision
    np.clip(lum, 0.0, 1.0, out=lum)
    return lum


def getY(lum):
    '''
    Get Y from luminance. Compuation detail:

    y[lum > 0.08] = ((y * 100.0 + 16.0) / 116.0) ** 3.0
    y[lum < 0.08] = y * 100.0 / 903.3

    Parameters
    ----------
    lum: 2D numpy.ndarray
        Input luminance data

    Returns
    -------
    y: 2D numpy.ndarray
        Converted to Y
    '''
    y = np.empty_like(lum)
    y[lum > 0.08] = ((lum[lum > 0.08] * 100.0 + 16.0) / 116.0) ** 3.0
    y[lum < 0.08] = lum[lum < 0.08] / 9.033
    # Clip the output due to float precision
    np.clip(y, 0.0, 1.0, out=y)
    return y
