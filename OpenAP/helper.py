import cv2
import numpy as np


def max_type(dtype):
    '''
    max_type(dtype(numpy.dtype))

    Get the maximum value of a given dtype. Note that for numpy.float32 based
    images, the maximum value is 1.0, not max of float32.

    dtype: numpy.dtype
    '''
    if dtype == np.float32 or dtype == np.float64:
        return 1.0
    else:
        return np.iinfo(dtype).max


def toGrayScale(img_data):
    '''
    toGrayScale(img_data(3D numpy.ndarray))

    Transform input image to gray scale. Note that we assume the input is
    in BGR.

    img_data: 3D numpy.ndarray
    '''
    return cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
