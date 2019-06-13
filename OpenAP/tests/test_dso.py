import cv2
import numpy as np
from OpenAP.background import gridBackgroundRemove
from OpenAP.correction import gammaCorrection, scurveRNC01, scurveRNC02
from OpenAP.frequency import aTrousTransform, iATrousTransform, starletKernel
from OpenAP.helper import convertTo
from OpenAP.histogram import applyLHE
from OpenAP.image import Image
from OpenAP.mask import globalSigmaMask


img_data = cv2.imread("./M51.tif", -1)
'''
# Before any processing
img_show_init = convertTo(img_data, np.uint16)
img_show_init = gammaCorrection(img_show_init, 0.5)
img_show_init = applyLHE(img_show_init, 20.0, (16, 16))
img_show_init = Image("Initial", img_show_init, ["B", "G", "R"])
img_show_init.saveTo("M51_init.jpg")
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
'''
img_starlet = []
# Wavelet
for i in range(3):
    tmp = aTrousTransform(img_no_bg[:, :, i], starletKernel,
                          max_level=5)
    img_starlet.append(tmp)
'''
'''
# Draw wavelet
for i in range(len(img_starlet[0])):
    fig = plt.figure(figsize=(30, 90))
    for j in range(3):
        ax = fig.add_subplot(3, 1, j + 1)
        med, std = np.median(img_starlet[j][i]), np.std(img_starlet[j][i])
        im = ax.imshow(img_starlet[j][i], cmap="jet", vmin=med-std,
                       vmax=med+std)
        plt.colorbar(im, ax=ax)
    plt.savefig("starlet_l" + str(i) + ".png")
    plt.close()
'''
'''
# Detail enhancement layer 1 2 3 4
for i in range(3):
    for j in [2]:
        img_starlet[i][j] = img_starlet[i][j] * 2.0
# Invert wavelet
img_istarlet = np.empty(img_data.shape)
for i in range(3):
    img_istarlet[:, :, i] = iATrousTransform(img_starlet[i], starletKernel)
'''
# Stretch and save to file
# img_save = convertTo(img_istarlet, np.uint16)
img_save = convertTo(img_no_bg, np.uint16)
img_save = gammaCorrection(img_save, 0.4)
img_save = applyLHE(img_save, 20.0, (16, 16))
img_save = scurveRNC01(img_save)
img_save = scurveRNC02(img_save)
# Save to file
img_save = Image("Final", img_save, ["B", "G", "R"])
img_save.saveTo("M51_final.jpg")
