import numpy as np


def max_type(dtype):
    '''
    max_type(dtype(numpy.dtype))

    Get the maximum value of a given dtype. Note that for numpy.float32 based
    images, the maximum value is 1.0, not max of float32.

    dtype: numpy.dtype
    '''
    if dtype == np.float32:
        return 1.0
    else:
        return np.iinfo(dtype).max
