import cv2
import numpy as np
from .helper import max_type


def starDetection(img_data, minThres, minRadius, minCirc, minInertia):
    '''
    starDetection(img_data(2D numpy.ndarray), minThres, minRadius(int),
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
