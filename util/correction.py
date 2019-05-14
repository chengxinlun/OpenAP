import numpy as np
from .helper import max_type


__all__ = ["gammaCorrection"]


def gammaCorrection(img_data, gamma):
    '''
    gammaCorrection(img_data(2 or 3D numpy.ndarray), gamma(float))

    Apply gamma correction to input image according to
        f(x) = (x / max_in_dtype) ** gamma * max_in_dtype

    img_data: 2 or 3D numpy.ndarray, the input image data
    gamma: float, the gamma value
    '''
    # Build up a look-up table for numpy.uint8 and uint16
    if img_data.dtype == np.uint8 or img_data.dtype == np.uint16:
        imax = max_type(img_data.dtype)
        lut = np.linspace(0, imax, imax + 1, dtype=np.float32) / imax
        lut = np.clip(np.power(lut, gamma) * imax, 0, imax)
        lut = lut.astype(img_data.dtype, copy=False)
        return np.take(lut, img_data)
    else:
        return np.clip(np.power(img_data, gamma), 0.0, 1.0)
