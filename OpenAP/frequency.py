# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy import signal
from skimage import restoration
from .helper import convertTo


__all__ = ['aTrousTransform', 'iATrousTransform', 'rlDeconvolve',
           'starletKernel']


def rlDeconvolve(img_data, psfFunction, psfParam, psfWidth=16, iterations=30,
                 newWidth=0.5):
    # skimage.restoration.richardson_lucy accepts float64 as input
    img_data_64 = convertTo(img_data, np.float64)
    # Set up PSF
    xx = np.linspace(-psfWidth, psfWidth, 2 * psfWidth + 1, dtype=np.float32)
    X, Y = np.meshgrid(xx, xx)
    pos = np.array([X.ravel(), Y.ravel()]).T
    psf = psfFunction(pos, 1.0, 0.0, 0.0, psfParam[0], psfParam[1],
                      psfParam[2], psfParam[3], 0.0).reshape(2 * psfWidth + 1,
                                                             2 * psfWidth + 1)
    # PSF for the deconvoluted image
    psf_new = psfFunction(pos, 1.0, 0.0, 0.0,
                          newWidth * newWidth * psfParam[0],
                          newWidth * newWidth * psfParam[1],
                          newWidth * newWidth * psfParam[2],
                          newWidth * newWidth * psfParam[3],
                          0.0).reshape(2 * psfWidth + 1, 2 * psfWidth + 1)
    # The PSF for deconvolution
    psf_deconv = restoration.richardson_lucy(psf, psf_new,
                                             iterations=iterations)
    # Deconvolve
    if img_data_64.ndim != 3:
        img_deconv = restoration.richardson_lucy(img_data_64, psf_deconv,
                                                 iterations=iterations)
    else:
        img_deconv = np.empty(img_data_64.shape)
        for i in range(img_data_64.shape[-1]):
            img_deconv[:, :, i] = restoration.richardson_lucy(
                img_data_64[:, :, i], psf_deconv, iterations=iterations)
    return img_deconv


def starletKernel(level):
    '''
    B3-spline kernel for starlet transformation. The scaling function is
    \phi(x) = 1/12 (|x-2|^3-4|x-1|^3+6|x|^3-4|x+1|^3+|x+2|^3)
    Thus the coeffcient for convolution is (1/16, 1/4, 3/8, 1/4, 1/16)

    Parameters
    ----------
    level: unsigned integer
        Level of the wavelet transformation. The distance between each
        coefficient is 2 ** level.

    Returns
    -------
    numpy.ndarray of shape (4 * 2 ** level + 1) and dtype == numpy.float32
        The kernel for convolution

    Raises
    ------
    None
    '''
    step = 2 ** level
    k_length = 4 * step + 1
    kernel = np.zeros((k_length), dtype=np.float64)
    kernel[0] = 1.0 / 16.0
    kernel[step] = 4.0 / 16.0
    kernel[2 * step] = 6.0 / 16.0
    kernel[3 * step] = 4.0 / 16.0
    kernel[-1] = 1.0 / 16.0
    return kernel


def _convValid(x, y):
    return np.convolve(x, y, 'valid')


def _convolve2DHDR(img_data, kernel):
    '''
    High dynamic range 2D convolve. Instead of FFT, this function use
    direct convolve, which is considerably slower.
    '''
    to_fill_length = int(np.floor(len(kernel) / 2))
    np_conv2d = np.vectorize(_convValid, signature='(n),(m)->(k)')
    # Row convolve
    tmp_row = np.pad(img_data, ((0, 0), (to_fill_length, to_fill_length)),
                     'symmetric')
    row_conv = np_conv2d(tmp_row, kernel)
    # Column convolve
    tmp_col = row_conv.T
    tmp_col = np.pad(tmp_col, ((0, 0), (to_fill_length, to_fill_length)),
                     'symmetric')
    col_conv = np_conv2d(tmp_col, kernel)
    return col_conv.T


def aTrousTransform(img_data, kernelFunc, max_level=5):
    '''
    Algorithm a trous (aka stationary wavelet transform) for 2D image.
    Convolution is taken care of through OpenCV. Assume kernel is separable
    in 2-D and the kernels are the same.

    Parameters
    ----------
    img_data: 2D numpy.ndarray
        Input image data. Automatically converted to numpy.float32.
    kernelFunc: function
        Function with level as input and convolution kernel as return
    max_level: unsigned integer, default 5
        Maximum level to wavelet transform. Starts with 0.

    Returns
    -------
    List of numpy.ndarray
        List of coefficent and approximation (w_0, ... w_{max_level-1},
        c_{max_level})

    Raises
    ------
    None
    '''
    if img_data.dtype != np.float64:
        img_in = convertTo(img_data, np.float64)
    else:
        img_in = img_data
    img_out = None
    res = []
    for i in range(max_level):
        kernel = kernelFunc(i)
        img_out = _convolve2DHDR(img_in, kernel)
        res.append(img_in - img_out)
        img_in = img_out
    res.append(img_out)
    return res


def iATrousTransform(coeff, kernelFunc):
    '''
    Inverse a Trous transform.

    Parameters
    ----------
    coeff: list of numpy.ndarray
        List of coefficients from aTrousTransform

    Returns
    -------
    2D numpy.ndarray
        Reconstructed image

    Raises
    ------
    None
    '''
    return np.sum(coeff, axis=0)
