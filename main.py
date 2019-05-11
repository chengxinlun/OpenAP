import cv2
import numpy as np
from image import Image
from util.correction import gammaCorrection
from util.helper import toGrayScale
from util.histogram import applyLHE, plotHist
from util.star import starDetection


# Read the image as it is
img_data = cv2.imread("../../Autosave.tif", -1)
# Construct Photo class and cast to CV_16U
img = Image("Test", img_data, ["B", "G", "R"])
img.convertTo(np.uint16, overwrite=True)
# Gamma correction
img_gamma = gammaCorrection(img.getData(), 0.5)
# Local histogram equalization
img_lhe = applyLHE(img_gamma, 10.0, (16, 16))
# DOG star detection
img_final = Image("Test Final", img_lhe, ["B", "G", "R"])
img_8 = img_final.convertTo(np.uint8, overwrite=False)
img_gray = toGrayScale(img_8)
kp = starDetection(img_gray, 20.0, 3, 0.8, 0.7)
print("Number of stars: " + str(len(kp)))
img_kp = cv2.drawKeypoints(img_8, kp, np.array([]), (0, 0, 255),
                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("stars.png", img_kp)
# Create a new image and save
# img_final.saveTo("../../Autosave_final.tif")
# img_final.saveTo("../../Autosave_final.png")
