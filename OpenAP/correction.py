# -*- coding: utf-8 -*-
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


def scurveRNC01(img_data):
    '''
    scurve1 in RNC-color correction. Adapted from c source file.
    Optimized with look-up table if the input is uint8 or uint16.

    More details, please visit:
        http://www.clarkvision.com/articles/astrophotography-rnc-color-stretch/
    '''
    # Build look-up table for numpy.uint8 or uint16
    zp = 5.0 / (1.0 + np.exp(2.1)) - 0.58
    op = 5.0 / (1.0 + np.exp(-2.9)) - 0.58 - zp
    if img_data.dtype == np.uint8 or img_data.dtype == np.uint16:
        imax = max_type(img_data.dtype)
        lut = np.linspace(0, imax, imax + 1, dtype=np.float32) / imax
        lut = 5.0 / (1.0 + np.exp(-5.0 * (lut - 0.42))) - 0.58
        lut = (lut - zp) / op * imax
        lut = lut.astype(img_data.dtype, copy=False)
        return np.take(lut, img_data)
    else:
        tmp = 5.0 / (1.0 + np.exp(-5.0 * (img_data - 0.42))) - 0.58
        tmp = (tmp - zp) / op
        return np.clip(tmp, 0.0, 1.0)


def scurveRNC02(img_data):
    '''
    scurve2 in RNC-color correction. Adapted from c source file.
    Optimized with look-up table if the input is uint8 or uint16.
    '''
    zp = 3.0 / (1.0 + np.exp(0.66)) - 0.78
    op = 3.0 / (1.0 + np.exp(-2.34)) - 0.78 - zp
    # Build look-up table for numpy.uint8 or uint16
    if img_data.dtype == np.uint8 or img_data.dtype == np.uint16:
        imax = max_type(img_data.dtype)
        lut = np.linspace(0, imax, imax + 1, dtype=np.float32) / imax
        lut = 3.0 / (1.0 + np.exp(-3.0 * (lut - 0.22))) - 0.78
        lut = (lut - zp) / op * imax
        lut = lut.astype(img_data.dtype, copy=False)
        return np.take(lut, img_data)
    else:
        tmp = 3.0 / (1.0 + np.exp(-3.0 * (img_data - 0.22))) - 0.78
        tmp = (tmp - zp) / op
        return np.clip(tmp, 0.0, 1.0)
