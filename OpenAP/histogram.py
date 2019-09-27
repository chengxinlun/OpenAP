# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
from .helper import max_type


__all__ = ["applyLHE"]


def applyLHE(img_data, clip_limit, grid_size):
    '''
    applyLHE(img_data(2 or 3 D numpy.ndarray, clip_limit(float),
    grid_size(tuple of int))

    Apply Local Histogram Equilization (LHE) with Contrast Limited
    Adaptive Histogram Equilization (CLAHE) in OpenCV.

    img_data: 2 or 3D numpy.ndarray, the image data
    clip_limit: float, maximum allowed contrast.
    grid_size: tuple of int, size of the grid
    '''
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    if img_data.ndim == 2:
        return clahe.apply(img_data)
    else:
        img_lhe = np.copy(img_data)
        for i in range(img_data.shape[-1]):
            img_lhe[:, :, i] = clahe.apply(img_data[:, :, i])
        return img_lhe
