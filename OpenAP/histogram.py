import cv2
import matplotlib.pyplot as plt
import numpy as np
from .helper import max_type


__all__ = ["calcHist", "plotHist"]


def calcHist(img_data):
    '''
    calcHist(img_data)

    Calculate the histrogram of input data

    img_data: 2 or 3 D numpy.ndarray, input image data
    '''
    hist_max = max_type(img_data.dtype)
    if img_data.ndim == 2:
        return cv2.calcHist([img_data], [0], None, [hist_max], [0, hist_max])
    else:
        hist = []
        for i in range(img_data.shape[-1]):
            hist.append(cv2.calcHist([img_data], [i], None, [hist_max],
                                     [0, hist_max]))
        return hist


def plotHist(img_data, cc_list):
    '''
    plotHist(img_data(2 or 3D numpy.ndarray), cc_list(list of str))

    Plot the histogram of input image

    img_data: 2 or 3D numpy.ndarray, the image data
    cc_list: list of str, name of color channel
    '''
    hist_max = max_type(img_data.dtype)
    if img_data.ndim == 2:
        plt.hist(img_data.ravel(), bins=256, range=(0, hist_max))
        plt.xlabel(cc_list[0])
    else:
        for i in range(img_data.shape[-1]):
            plt.hist(img_data[:, :, i].ravel(), bins=256, range=(0, hist_max),
                     alpha=0.5, label=cc_list[i], color=cc_list[i])
        plt.legend(loc='upper right')
    plt.show()
    plt.close()


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
