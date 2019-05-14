from numba import jit
import numpy as np


__all__ = ["gaussian2D"]


@jit
def gaussian2D(x, amp, mu_x, mu_y, sigma_xx, sigma_xy, sigma_yx, sigma_yy,
               offset):
    '''
    gaussian2D(x(numpy.ndarray), amp(float), mu(1D numpy.ndarray),
               sigma(2D numpy.ndarray))

    Calculate 2D gaussian, Input array can be in any shape as long as it
    satisfies the requirement below. Return array will be in the same shape
    wihtout the last dimension.

    x: numpy.ndarray, x[..., 0] = X, x[..., 1] = Y
    amp: float, amplitude
    mu_x, mu_y: float, the center of the gaussian
    sigma_xx, sigma_xy, sigma_yx, sigma_yy: float, sigma matrix
    offset: float, offset
    '''
    mu = np.array([mu_x, mu_y])
    sigma = np.array([[sigma_xx, sigma_xy], [sigma_yx, sigma_yy]])
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    effi = np.sqrt((2.0 * np.pi) ** 2.0 * sigma_det)
    # sigma^{kl}*(x-mu)_{k}*(x-mu)_{l}
    fac = np.einsum('...k,kl,...l->...', x - mu, sigma_inv, x - mu)
    return amp * np.exp(-fac / 2.0) / effi + offset
