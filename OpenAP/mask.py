# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .color import getLuminance
from .helper import max_type, convertTo


__all__ = ["dogStarMask", "gbDog", "globalSigmaMask"]


def dogStarMask(img_data, sigma01, sigma02, dtype=np.uint8, binary=True,
                thres=150):
    '''
    dogStarMask(img_data(2D or 3D numpy.ndarray), sigma01(float),
                simga02(float), thres(int or float))

    Difference of Gaussians (DOG) star detection. DOG can extract darker stars
    than SimpleBlob. However, the quality of stars is not suitable for star
    fitting.

    img_data: 2D or 3D numpy.ndarray, input image data.
    sigma01: float, sigma of the first Gaussian blur.
    sigma02: float, sigma of the second Gaussian blur.
    dtype: numpy.dtype, default numpy.uint8. dtype of the output.
    binary: boolean, default True. If the output is binary or not.
    thres: same dtype with img_data, threshold to mask. Value above this
           threshold will be set to max dtype and those below to 0.
    '''
    gb01 = gbDog(img_data, sigma01)
    gb02 = gbDog(img_data, sigma02)
    dog = gb02 - gb01
    # Convert to mask
    if binary:
        dog[dog < thres] = 0
        dog[dog >= thres] = max_type(dog.dtype)
    return convertTo(dog, dtype)


def gbDog(img_data, sigma):
    '''
    gbDog(img_data(2D or 3D numpy.ndarray), sigma(float))

    Gaussian blur for DOG.
    '''
    return cv2.GaussianBlur(img_data, (0, 0), sigma)


def globalSigmaMask(img_data, thres, bgrWeight, dtype=np.uint8, binary=True):
    '''
    Global sigma mask

    For colored images:
    Convert image to Luminosity and mask everything under med + thres * std

    For monocolor images:
    Mask everything under med + thres * std

    Parameters
    ----------
    img_data: 3D or 2D numpy.ndarray
        The input image data

    thres: float
        Threshold for sigma clipping

    bgrWeight: 1D numpy.ndarray with shape == (3,)
        Only applicable for colored images, the weight for luminance

    dtype: numpy.dtype, default numpy.uint8
        dtype of return

    binary: boolean, default True
        Whether the mask is binary or not
    '''
    if img_data.ndim == 3:
        img_lum = getLuminance(img_data, bgrWeight)
    else:
        img_lum = img_data
    img_mask = np.zeros(img_lum.shape, dtype=dtype)
    img_clip = np.median(img_lum) + thres * np.std(img_lum)
    if binary:
        img_mask[img_lum > img_clip] = max_type(dtype)
    else:
        img_mask[img_lum > img_clip] = img_lum[img_lum > img_clip]
    return img_mask
