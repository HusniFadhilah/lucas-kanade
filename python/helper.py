import numpy as np
import cv2,os
import scipy.io as sio
from matplotlib import pyplot as plt
import skimage.feature


dirname = os.path.dirname(__file__)
datadir = os.path.join(dirname, '../data')
resultdir = os.path.join(dirname, '../results3')