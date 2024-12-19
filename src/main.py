import matplotlib.image as mpimg
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
from utils import *



if __name__ == "__main__":
    #LoadImage
    img_src = cv2.imread('input/14.jpg')
    w, h, d = original_shape = tuple(img_src.shape)
    assert d == 3

    # Number of samples per component
    n_samples = w*h
    #Number of sets of training samples
    num_patches=100;


    samples, imtest=createData(img_src, n_samples)
    gmm, dpgmm=train(num_patches, samples ,n_samples,w,h)
    lab1,lab2=test(samples, gmm, dpgmm)
    seg1=segmented(img_src,samples,lab1, 7, w, h)
    seg2=segmented(img_src,samples, lab2,7, w, h)

    
    img_src = mpimg.imread('input/14.jpg') 


    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
    axs[0].imshow(img_src)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(seg1)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.imsave('output/14.png', seg1)

