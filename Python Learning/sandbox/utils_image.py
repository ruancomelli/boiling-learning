import tensorflow

import utils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd


img2D = np.random.rand(5, 5)
img = np.random.rand(5, 5, 4)
print(img)
print('>> img2D')
print(img2D)
print('>> Cropped img2D')
print(utils.image.crop(img2D, lims=[(1, 3), (3, 4)]))
print(utils.image.crop(img2D, x_min=1, x_max=3, y_min=3, y_max=4))
print(utils.image.crop(img2D, x_lim=(1,3), y_lim=(3,4)))

    