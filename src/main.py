import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import math
import matplotlib as mpl 
import itertools
from matplotlib import colors as mcolors
import itertools
from itertools import chain
import matplotlib.cm as cm 
import utils


if __name__ == "__main__":
    img_src = cv2.imread("input/10.jpg")
    w, h, d = original_shape = tuple(img_src.shape)
    assert d == 3

    n_samples, num_patches = w*h, 100
    print(w, h, d, n_samples, num_patches)

    samples, imtest = utils.create_data(img_src, n_samples)

    gmm, dpgmm=train(num_patches, samples ,n_samples,w,h)
#prepimage(imtest, num_patches)

    
    pass