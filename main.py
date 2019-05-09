import cv2
import matplotlib.pyplot as plt
import numpy as np
from photo import Photo
from util.correction import gammaCorrection
from util.histogram import applyLHE, calcHist, plotHist


# Read the image as it is
img = cv2.imread("../../Autosave.tif", -1)
# Construct Photo class and cast to CV_16U
pho = Photo("Test", img, ["B", "G", "R"])
pho.convertTo(np.uint16, overwrite=True)
# Gamma correction
img_gamma = gammaCorrection(pho.getData(), 0.5)
# Local histogram equalization
img_lhe = applyLHE(img_gamma, 20.0, (16, 16))
cv2.imwrite("stretched.tif", img_lhe)
plotHist(img_lhe, pho.getColorChannelName())
# Convert from CV_16U to CV_8U
img_lhe = np.uint8(img_lhe / 256)
cv2.imwrite("stretched.jpg", img_lhe)
