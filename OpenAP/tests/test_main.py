# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pywt
from OpenAP.image import Image
from OpenAP.correction import gammaCorrection
from OpenAP.frequency import rlDeconvolve
from OpenAP.helper import convertTo, toGrayScale
from OpenAP.histogram import applyLHE, plotHist
from OpenAP.logging import setupLogger
from OpenAP.mask import dogStarMask
from OpenAP.psf import gaussian2D
from OpenAP.star import blobStarDetection, starFitting, starSizeReduction


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
# Set up logger
setupLogger("openap.log", True)
# Read the image as it is
img_data = cv2.imread("MilkyWay.tif", -1)
# Construct Photo class and cast to CV_16U
img_class_init = Image("MilkyWay", img_data, ["B", "G", "R"])
img_class_init.convertTo(np.uint16, overwrite=True)
# -----------------------------------------------------------------------------
# Linear transformations
# ----------------------------------------------------------------------------
# Deconvolution is not recommended due to lack of deringring algorithm
'''
# Deconvolution
#     01: star detection
img_8 = convertTo(img_class_init.getData(), np.uint8)
img_gray = toGrayScale(img_8)
kp = blobStarDetection(img_gray, 10.0, 2, 0.8, 0.7)
print("Number of stars: " + str(len(kp)))
img_gamma = gammaCorrection(img_class_init.getData(), 0.5)  # For plotting only
img_lhe = applyLHE(img_gamma, 10.0, (16, 16))  # For plotting only
img_lhe = convertTo(img_lhe, np.uint8)
img_kp = cv2.drawKeypoints(img_lhe, kp, np.array([]), (0, 0, 255),
                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("MilkyWay_stars.png", img_kp)
#     02: PSF fitting
sigma_matrix, residual = starFitting(img_class_init.getData()[:, :, 0], kp,
                                     gaussian2D)  # PSF Fitting
print("Number of stars fitted: " + str(len(sigma_matrix[0])))
sigma_matrix = np.median(np.array(sigma_matrix), axis=1)
print("Sigma matrix: " + str(sigma_matrix))
#     03: deconvolution
img_deconv = rlDeconvolve(img_class_init.getData(), gaussian2D, sigma_matrix,
                          psfWidth=5)
np.save("deconv.npy", img_deconv)
img_deconv = convertTo(img_deconv, np.uint16)
'''
# Stationary wavelet transform (Algorithme a Torus)
wavelet = pywt.Wavelet('db2')
level = 5
#     01: pad the array to a multiple of 2 ** level
img_float = img_class_init.convertTo(np.float64, overwrite=False)
pad = []
for i in range(2):
    multi = np.ceil(img_float.shape[i] / (2 ** level))
    pad.append(int(multi * (2 ** level) - img_float.shape[i]))
img_padded = np.pad(img_float, ((0, pad[0]), (0, pad[1]), (0, 0)), 'symmetric')
img_swt = pywt.swt2(img_padded[:, :, 0], wavelet, level)
img_iswt = pywt.iswt2(img_swt, wavelet)
img_iswt = img_iswt[:img_float.shape[0], :img_float.shape[1]]
cv2.imwrite("iswt.tif", convertTo(img_iswt, np.uint16))
# -----------------------------------------------------------------------------
# Non-linear transformation
# -----------------------------------------------------------------------------
'''
# Step 1: Contrast adjustment
#     01: Gamma correction
img_nl01 = gammaCorrection(img_l_final, 0.5)
#     02: Local histogram equality
img_nl01 = applyLHE(img_nl01, 10.0, (16, 16))
# Step 2: Star size reduction
#     01: get a star mask
img_gray = toGrayScale(img_nl01)
star_mask = dogStarMask(img_gray, 3, 4.5, 125)
#     02: morphological transform
erosion_mask = np.array([[0, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 0]], dtype=np.uint8)
img_nl02 = starSizeReduction(img_nl01, star_mask, erosion_mask,
                             ratio=0.5, eroIter=2, dilIter=3)
'''
