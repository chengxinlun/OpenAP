import numpy as np
from skimage import restoration
from .helper import convertTo


__all__ = ['rlDeconvolve']


def rlDeconvolve(img_data, psfFunction, psfParam, psfWidth=16, iterations=30):
    # skimage.restoration.richardson_lucy accepts float64 as input
    img_data_64 = convertTo(img_data, np.float64)
    # Set up PSF
    xx = np.linspace(-psfWidth, psfWidth, 2 * psfWidth + 1, dtype=np.float32)
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
