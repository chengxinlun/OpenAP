# -*- coding: utf-8 -*-
import cv2
import logging
import numpy as np
from scipy.optimize import curve_fit
from .helper import max_type, toGrayScale, convertTo


__all__ = ["blobStarDetection", "_starFitting", "starFitting",
           "starSizeReduction"]


def starSizeReduction(img_data, starMask, erosionMask, ratio=0.5, eroIter=1,
                      dilIter=3):
    '''
    starSizeReduction(img_data(2D or 3D numpy.ndarray), starMask(2D np.uint8
                      numpy.ndarray), erosionMask(2D np.uint8 numpy.ndarray),
                      ratio(float), eroIter(int), dilIter(int))

    Reduce star size with morphological transformation. Star mask dilation is
    also performed if necessary. Output is a linear combination of original
    image and star-size-reduced image.

    img_data: 2D or 3D numpy.ndarray, input image data.
    starMask: 2D numpy.uint8 array, input star mask.
    erosionMask: 2D numpy.uint8 array, erosion and dilation mask.
    ratio: float, default 0.5. Coefficient for linear combination. Should be
           less than 1.
    eroIter: int, default 1. Number of iterations of erosion.
    dilIter: int, default 3. Number of iterations of star mask dilation.
    '''
    # Dilate star mask
    if dilIter != 0:
        star_mask = cv2.dilate(starMask, erosionMask, iterations=dilIter)
    else:
        star_mask = starMask
    # Extract stars
    img_star_only = cv2.bitwise_and(img_data, img_data, mask=star_mask)
    # Erosion
    img_star_only_reduced = cv2.erode(img_star_only, erosionMask,
                                      iterations=eroIter)
    # Get black in the eroded image
    mask_to_fill = toGrayScale(img_star_only_reduced)
    mask_to_fill = convertTo(mask_to_fill, np.uint8)
    mask_to_fill[mask_to_fill != 0] = 255
    img_to_fill_1 = cv2.bitwise_and(img_data, img_data, mask=mask_to_fill)
    mask_to_fill = cv2.bitwise_not(mask_to_fill)
    img_to_fill_2 = cv2.bitwise_and(img_data, img_data, mask=mask_to_fill)
    img_part = cv2.addWeighted(img_star_only_reduced, ratio, img_to_fill_1,
                               1.0 - ratio, 0.0)
    return cv2.add(img_part, img_to_fill_2)


def blobStarDetection(img_data, minThres, minRadius, minCirc, minInertia):
    '''
    blobstarDetection(img_data(2D numpy.ndarray), minThres, minRadius(int),
                      minCirc(float), minInertia(float))


    Star detection with SimpleBlobDetector in OpenCV. Max threshold is assumed
    to be 255(uint8) or 65535(uint16). Convexity is not enabled.

    img_data: 2D numpy.ndarray, input gray-scale image
    minThres: float, minimum threshold
    minRadius: float, minimum radius
    minCirc: float, minimum circularity. Should be greater than 0 and less than
             1. Circle == 1, square = 0.785, ...
    minInertia: float, minimum inertia. Should be greater than 0 and less than
                1. Perfect circle == 1, ellispse < 1, line == 0, ...
    '''
    # Set up parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = minThres
    params.maxThreshold = 225
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = np.pi * minRadius * minRadius
    params.filterByConvexity = False
    params.filterByCircularity = True
    params.minCircularity = minCirc
    params.filterByInertia = True
    params.minInertiaRatio = minInertia
    # Create detector
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(img_data)


def _starFitting(img_data, keypoint, fitFunction, fitRadius):
    '''
    _starFitting(img_data(2D numpy.ndarray), keypoint(cv2.Point2f),
                 fitRadius(int))

    Fit individual star.
    '''
    coord = np.int32(np.round(keypoint.pt))
    fit_data = img_data[coord[1] - fitRadius: coord[1] + fitRadius + 1,
                        coord[0] - fitRadius: coord[0] + fitRadius + 1]
    fit_data = fit_data.astype(np.float32)
    if np.max(fit_data) > 0.9 * max_type(img_data.dtype):
        raise ValueError("Star " + str(coord) + " saturated. Skip fitting.")
    xx = np.linspace(-fitRadius, fitRadius, 2 * fitRadius + 1,
                     dtype=np.float32)
    X, Y = np.meshgrid(xx, xx)
    pos = np.array([X.ravel(), Y.ravel()]).T
    ig = [np.max(fit_data), 0.0, 0.0, keypoint.size * 0.1, 0.0, 0.0,
          keypoint.size * 0.1, 0.0]
    try:
        popt, pcov = curve_fit(fitFunction, pos, fit_data.ravel(), p0=ig,
                               maxfev=200000)
    except Exception as reason:
        raise RuntimeError("Star " + str(coord) + " failed to fit: " +
                           str(reason))
    fit_res = fitFunction(pos, *popt).reshape(2 * fitRadius + 1,
                                              2 * fitRadius + 1)
    res = popt
    res[1] = res[1] + coord[1]
    res[2] = res[2] + coord[0]
    return [res, fit_data - fit_res]


def starFitting(img_data, keypointList, fitFunction, fitRadius=8):
    '''
    starFitting(img_data(2D numpy.ndarray), keypointList(list of cv2.Point2f),
                fitFunction(function), fitRadius(int))
    Fit stars in keypointList with fitFunction

    img_data: 2D numpy.ndarray, input image data
    keypointList: list of cv2.Point2f, list of stars as keypoints
    fitFunction: function, function for star (PSF). This function must have
                 the input of (x(2D numpy.ndarray, x[:, 0] == X, x[:, 1] == Y),
                 amplitude(float), x0(float), y0(float),
                 g_00, g_01, g_10, g_11 (coefficients for xx, xy, yx, yy))
    fitRadius: int, defualt 8. Radius to fit
    '''
    np.seterr("raise")
    g_mu_nu = [[], [], [], []]
    residual = []
    for each in keypointList:
        try:
            res = _starFitting(img_data, each, fitFunction, fitRadius)
        except Exception as exce:
            logger = logging.getLogger(__name__)
            logger.warning(str(exce))
            continue
        residual.append(res[1])
        for i in range(4):
            g_mu_nu[i].append(res[0][i + 3])
    return g_mu_nu, residual
