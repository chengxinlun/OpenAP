import logging
import numpy as np


__all__ = ["setupLogger"]


def setupLogger(fileName=None, printError=True):
    # Convert numpy error/warning message to real error/warning
    np.seterr(all="warn")
    # Set up root logger
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        "[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s")
    # Set up file logger
    if fileName is not None:
        fh = logging.FileHandler(fileName)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    # Set up stdio logger
    if printError:
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)
