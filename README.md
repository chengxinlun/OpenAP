## Astrophotography Post Processing with OpenCV
#### Dependencies
* Python (Python 3 recommended, not tested on Python 2)
* Matplotlib
* Numba
* Numpy
* OpenCV
* Scipy
* Scikit-image

## Current Functionalities
* 8-bit, 16-bit, 32-bit image conversion
* Checking histogram
* Deconvolution
* Gamma correction
* Local Histogram Equilization
* Logging
* Simple star detection
* Star (PSF) fitting with 2D Gaussian (more PSF models in development)

## Example
```python
import cv2
import numpy as np
from OpenAP.image import Image
from OpenAP.util.correction import gammaCorrection
from OpenAP.util.frequency import rlDeconvolve
from OpenAP.util.helper import convertTo, toGrayScale
from OpenAP.util.histogram import applyLHE
from OpenAP.logging import setupLogger
from OpenAP.util.psf import gaussian2D
from OpenAP.util.star import starDetection, starFitting


# Set up logger
setupLogger("openap.log", True)
# Read the image as it is
img_data = cv2.imread("MilkyWay.tif", -1)
# Construct Photo class and cast to CV_16U
img_class_init = Image("MilkyWay", img_data, ["B", "G", "R"])
img_class_init.convertTo(np.uint16, overwrite=True)
# Gamma correction
img_gamma = gammaCorrection(img_class_init.getData(), 0.5)
# Local histogram equalization
img_lhe = applyLHE(img_gamma, 10.0, (16, 16))
# Star detection
img_8 = convertTo(img_lhe, np.uint8)
img_gray = toGrayScale(img_8)
kp = starDetection(img_gray, 25.0, 2, 0.8, 0.7)
print("Number of stars: " + str(len(kp)))
img_kp = cv2.drawKeypoints(img_8, kp, np.array([]), (0, 0, 255),
                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("MilkyWay_stars.png", img_kp)
cv2.imwrite("MilkyWay_gamma_lhe.png", img_8)
# 2D Gaussian fitting
sigma_matrix, residual = starFitting(img_lhe[:, :, 0], kp, gaussian2D)
print("Number of stars fitted: " + str(len(sigma_matrix[0])))
# Deconvolution
sigma_matrix = np.array(sigma_matrix)
img_deconv = rlDeconvolve(img_lhe, gaussian2D,
                          np.median(sigma_matrix, axis=1))
# Create a new image and save
img_deconv_8 = Image("Deconv", img_deconv, ["B", "G", "R"])
img_deconv_8.saveTo("MilkyWay_gamma_lhe_deconv.png")
```
