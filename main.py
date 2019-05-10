import cv2
import matplotlib.pyplot as plt
import numpy as np
from image import Image
from util.correction import gammaCorrection
from util.histogram import applyLHE, calcHist, plotHist


# Read the image as it is
img_data = cv2.imread("../../Autosave.tif", -1)
# Construct Photo class and cast to CV_16U
img = Image("Test", img_data, ["B", "G", "R"])
img.convertTo(np.uint16, overwrite=True)
# Gamma correction
img_gamma = gammaCorrection(img.getData(), 0.5)
# Local histogram equalization
img_lhe = applyLHE(img_gamma, 10.0, (16, 16))
cv2.imwrite("stretched.tif", img_lhe)
plotHist(img_lhe, img.getColorChannelName())
# Create a new image and save
img_final = Image("Test Final", img_lhe, ["B", "G", "R"])
img_final.saveTo("../../Autosave_final.tif")
img_final.saveTo("../../Autosave_final.png")
