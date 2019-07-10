# -*- coding: utf-8 -*-
import numpy as np
from skimage import restoration
from .helper import convertTo


__all__ = ['aTrousTransform', 'iATrousTransform', 'rlDeconvolve',
           'starletKernel']


def rlDeconvolve(img_data, psfFunction, psfParam, psfWidth=16, iterations=30):
    # skimage.restoration.richardson_lucy accepts float64 as input
    img_data_64 = convertTo(img_data, np.float64)
    # Set up PSF
    xx = np.linspace(-psfWidth, psfWidth, 2 * psfWidth + 1, dtype=np.float64)
    X, Y = np.meshgrid(xx, xx)
    pos = np.array([X.ravel(), Y.ravel()]).T
    psf = psfFunction(pos, 1.0, 0.0, 0.0, psfParam[0], psfParam[1],
                      psfParam[2], psfParam[3], 0.0).reshape(2 * psfWidth + 1,
                                                             2 * psfWidth + 1)
    # Deconvolve
    if img_data_64.ndim != 3:
        img_deconv = restoration.richardson_lucy(img_data_64, psf,
                                                 iterations=iterations)
    else:
        img_deconv = np.empty(img_data_64.shape)
        for i in range(img_data_64.shape[-1]):
            img_deconv[:, :, i] = restoration.richardson_lucy(
                img_data_64[:, :, i], psf, iterations=iterations)
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
    direct convolve, which is considerably slower. However, it is still
    suprisingly fast.
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


def msLinearTransform(img_data, img_mask, maxLevel, tList, iList, sList,
                      transFunc=[aTrousTransform, iATrousTransform],
                      kernelFunc=starletKernel):
    '''
    Multi-scale Linear Transform
    For noise reduction purpose, adapted from:
        Starck J-L, Bijaoui A, Murtagh F (1995) Multiresolution support applied
        to image filtering and deconvolution. CVGIP: Graph Models Image Process
        57: 420-431

    Parameters
    ----------
    img_data: 2D numpy.ndarray
        Input image data. For colored image, extract its luminance channel and
        then feed into this function
    img_mask: 2D numpy.ndarray
        Input image mask
    maxLevel: int
        Maximum level of decomposition
    tList: numpy.array of float, length == maxLevel
        List of threshold for each level, if filtering is not needed, use 0.0
    iList: list of int, length == maxLevel
        List of iterations for each level, if filtering is not needed for
        certain level, use 0
    sList: list of float, length == maxLevel, every value between 0.0 and 1.0
        List of strength for each level, 0.0 for no modification, 1.0 for
        denoise data only, value in between for linear combination of both
    transFunc: list of functions, length == 2
        [Transform function, inverse transform function]
    kernelFunc: function
        Kernel function for wavelet transform

    Returns
    -------
    img_recon: 2D numpy.ndarray, same shape as img_data
        Denoised image

    Raises
    ------
    ValueError
    '''
    if len(tList) != maxLevel or len(iList) != maxLevel:
        raise ValueError("tList and iList must have length of maxLevel")
    if img_data.ndim != 2:
        raise TypeError("img_data must be 2D. For colored image, extract" +
                        " its luminance channel")
    # Cast everything to everything to float64
    img_data_64 = convertTo(img_data, np.float64)
    img_mask_64 = convertTo(img_data, np.float64)
    # Estimate the relation between image space and frequency space
    # TODO: support look-up table for given transformation and kernel
    img_test = np.random.normal(0.0, 1.0, (2048, 2048))
    img_test_swt = transFunc[0](img_test, kernelFunc, maxLevel)
    noise_sigma_coeff = np.std(img_test_swt[:-1], axis=(1, 2))
    # Estimate noise of the image and determine filtering threshold
    img_swt = transFunc[0](img_data_64, kernelFunc, maxLevel)
    img_noise_sigma = np.std(img_swt[0]) / noise_sigma_coeff[0] * \
        noise_sigma_coeff
    print(img_noise_sigma[0] / noise_sigma_coeff[0])
    thres_list = tList * img_noise_sigma
    # Iteration
    n = 0
    sol = np.zeros_like(img_data_64)
    err = np.zeros_like(img_data_64)
    while n < np.max(iList):
        err = img_data_64 - sol
        err_swt = transFunc[0](err, kernelFunc, maxLevel)
        # Filter each level except residual
        for i in range(maxLevel):
            if n < iList[i]:
                tf = np.logical_and(err_swt[i] > -thres_list[i],
                                    err_swt[i] < thres_list[i])
                err_swt[i][tf] *= (1.0 - img_mask_64[tf] * sList[i])
        # Reconstruct
        err_iswt = transFunc[1](err_swt, kernelFunc)
        sol += err_iswt
        n += 1
    return sol
