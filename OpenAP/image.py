import cv2
import numpy as np
from .helper import *


__all__ = ["Image"]


class Image(object):
    def __init__(self, name, img_data, cc_name):
        '''
        __init__(name(str), img_data(numpy.ndarray, 2 or 3 D),
        cc_name(list of str))

        Construct Photo class for furhter processing. The pixel size and number
        of color channels are inferred from input data.

        name: str, a simple description
        img_data: 2 or 3 D numpy.ndarray, the pixel data
        cc_name: list of str, name of each color channel
        '''
        self.name = name
        self.data = img_data
        if img_data.ndim == 2:
            self.pix_size = img_data.shape
            self.color_channel = 1
            self.color = cc_name
        elif img_data.ndim == 3:
            self.pix_size = img_data.shape[:-1]
            self.color_channel = img_data.shape[-1]
            self.color = cc_name

    def __str__(self):
        return self.name + "\n" + "Size: " + str(self.pix_size) + "\n" +\
            "Color channel: " + str(self.color)

    def getData(self):
        return self.data

    def getSize(self):
        return self.pix_size

    def getColorChannelNumber(self):
        return self.color_channel

    def getColorChannelName(self):
        return self.color

    def convertTo(self, dtype, overwrite=False):
        '''
        convertTo(dtype(numpy.dtype), overwrite(boolean))

        Convert image data to a given dtype. Since OpenCV can only handle uint8
        or uint16 data, it is necessary to apply if the input data is directly
        from DSS.

        dtype: numpy.dtype, dtype to cast to. Usually numpy.uint8 or
               numpy.uint16. Seldomly, it can be numpy.float32.
        overwrite: boolean, whether overwrite data or not.
        '''
        tmp = convertTo(self.data, dtype)
        if overwrite:
            self.data = tmp
        else:
            return tmp

    def saveTo(self, filename, img_format=None):
        '''
        saveTo(filename(str), img_format(str))

        Save the image to certain format with given file name. Image format
        can be inferred from file name suffix.

        filename: str, file name to save to.
        img_format: str, default None. Image format. If None, it will be
                    inferred from filename.
        '''
        if img_format is None:
            imf = filename.rsplit('.', 1)[-1]
        else:
            imf = img_format
        if imf in ["jpg", "jpeg", "png", "gif", "bmp"]:
            img_save = self.convertTo(np.uint8, overwrite=False)
        else:
            img_save = self.data
        cv2.imwrite(filename, img_save)
