import cv2
import numpy as np


__all__ = ["max_type", "toGrayScale", "convertTo"]

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


def convertTo(img_data, dtype):
    '''
    convertTo(img_data(2D or 3D numpy.ndarray)dtype(numpy.dtype))

    Convert image data to a given dtype. Since OpenCV can only handle uint8
    or uint16 data, it is necessary to apply if the input data is directly
    from DSS.

    img_data: 2D or 3D numpy.ndarray, input image data
    dtype: numpy.dtype, dtype to cast to. Usually numpy.uint8 or
           numpy.uint16. Seldomly, it can be numpy.float32.
    '''
    if dtype != img_data.dtype:
        tmp = dtype(img_data / max_type(img_data.dtype) *
                    max_type(dtype))
    else:
        tmp = img_data
    return tmp
