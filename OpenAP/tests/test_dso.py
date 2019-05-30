import cv2
import numpy as np
from OpenAP.background import gridBackgroundRemove
from OpenAP.correction import gammaCorrection
from OpenAP.image import Image
from OpenAP.helper import convertTo, toGrayScale
from OpenAP.histogram import applyLHE


img_data = cv2.imread("./M51.tif", -1)
# Before any processing
img_show_init = convertTo(img_data, np.uint16)
img_show_init = gammaCorrection(img_show_init, 0.5)
img_show_init = applyLHE(img_show_init, 20.0, (16, 16))
img_show_init = Image("Initial", img_show_init, ["B", "G", "R"])
img_show_init.saveTo("M51_init.jpg")
# Luminosity mask
img_8 = convertTo(img_data, np.uint8)
img_gray = toGrayScale(img_8)
img_mask = np.zeros_like(img_gray)
img_clip = np.median(img_gray)
img_mask[img_gray > img_clip] = 255
# Make mask larger
img_mask = cv2.GaussianBlur(img_mask, (13, 13), 0)
img_mask[img_mask > 0] = 255
# Invert mask
img_mask = cv2.bitwise_not(img_mask)
# Grid background removal
img_no_bg = gridBackgroundRemove(img_data, 64, doMask=img_mask,
                                 infillLength=31, smoothSigma=5)
# Stretch and save to file
img_no_bg = convertTo(img_no_bg, np.uint16)
img_no_bg = gammaCorrection(img_no_bg, 0.5)
img_no_bg = applyLHE(img_no_bg, 20.0, (16, 16))
# Save to file
img_save02 = Image("Background subtraction", img_no_bg, ["B", "G", "R"])
img_save02.saveTo("M51_no_bg.jpg")
