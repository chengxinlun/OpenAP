# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .helper import max_type, convertTo


__all__ = ["dogStarMask", "gbDog"]


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
