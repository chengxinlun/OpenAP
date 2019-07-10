import cv2
import numpy as np
import matplotlib.pyplot as plt
from OpenAP.background import gridBackgroundRemove
from OpenAP.color import response_Bradford, white_D65, white_D50, \
    chromaticAdaptation, colorConvert, convert_sRGB2XYZ
from OpenAP.correction import gammaCorrection, scurveRNC01, scurveRNC02
from OpenAP.frequency import msLinearTransform, aTrousTransform, starletKernel
from OpenAP.helper import convertTo
from OpenAP.histogram import applyLHE
from OpenAP.image import Image
from OpenAP.mask import globalSigmaMask


img_data = cv2.imread("./M51.tif", -1)[:, :, ::-1]
img_data_xyz_d65 = colorConvert(img_data, convert_sRGB2XYZ)
img_data_xyz_d50 = chromaticAdaptation(img_data_xyz_d65, white_D65, white_D50,
                                       response_Bradford)
'''
# Luminosity mask
img_mask = globalSigmaMask(img_data, 1.5, np.array([1.0, 1.0, 1.0]))
# Make mask larger
img_mask = cv2.GaussianBlur(img_mask, (5, 5), 0)
img_mask[img_mask > 0] = 255
# Invert mask
img_mask = cv2.bitwise_not(img_mask)
cv2.imwrite("M51_mask.jpg", img_mask)
# Grid background removal
img_no_bg = gridBackgroundRemove(img_data, 64, doMask=img_mask,
                                 infillLength=31, smoothSigma=5)
plt.hist(img_no_bg[1800:1850, 2530:2560, 0].ravel(), bins=50, alpha=0.5)
plt.hist(img_no_bg[1800:1850, 2530:2560, 1].ravel(), bins=50, alpha=0.5)
plt.hist(img_no_bg[1800:1850, 2530:2560, 2].ravel(), bins=50, alpha=0.5)
plt.show()
img_save = convertTo(img_no_bg, np.uint16)
img_save = gammaCorrection(img_save, 0.4)
img_save = applyLHE(img_save, 20.0, (16, 16))
img_save = scurveRNC01(img_save)
img_save = scurveRNC02(img_save)
plt.imshow(convertTo(img_save[1800:1850, 2530:2560], np.uint8))
plt.show()
# Extract luminance
img_no_bg = convertTo(img_no_bg, np.float32)
img_lum = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2Lab)
# Without denoising
img_no_bg = cv2.cvtColor(img_lum, cv2.COLOR_Lab2BGR)
# Denoise
img_lum_mask = np.ones(img_data.shape[:-1], dtype=np.float64)
tmp = msLinearTransform(img_lum[0], img_lum_mask, 5,
                        np.array([3.0, 2.0, 1.0, 0.5, 0.0]),
                        [3, 2, 1, 1, 0], [0.5, 0.5, 0.5, 0.5, 0.5])
img_lum[0] = tmp.astype(np.float32)
# Stretch and save to file
img_no_noise = cv2.cvtColor(img_lum, cv2.COLOR_Lab2BGR)
img_save = convertTo(img_no_noise, np.uint16)
img_save = gammaCorrection(img_save, 0.4)
img_save = applyLHE(img_save, 20.0, (16, 16))
img_save = scurveRNC01(img_save)
img_save = scurveRNC02(img_save)
# Save to file
img_save = Image("Final", img_save, ["L"])
img_save.saveTo("M51_final.jpg")
'''
