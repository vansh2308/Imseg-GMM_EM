import numpy as np
import cv2
import itertools
import time
from skimage import feature
from sklearn import mixture
from sklearn import preprocessing
from sklearn.utils import shuffle
from itertools import chain


color_iter = itertools.cycle(['navy', 'red', 'cornflowerblue', 'gold', 'darkorange','b','cyan'])
_color_codes = {
    1: (171,166, 27),
    2: (112, 26, 91,),
    3: (61, 42,   61), 
    4: (19, 118, 140),
    5: (227, 25, 227),
    6: (139, 69,   19),
    7: (56, 161,  48)
}
d = 3


def create_data(image, n_samples):
    num_pts, radius = 24, 8
    img_src = cv2.GaussianBlur(image,(5,5),0)

    # Projecting to CIELAB space 
    imtest=cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB) 

    # Captures local texture 
    img_gray= cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(img_gray, num_pts, radius, method="uniform")
    lbp = np.reshape(lbp,(n_samples,1))

    imtest= np.reshape(imtest, (n_samples, d))
    data=np.column_stack((imtest, lbp))

    data = preprocessing.normalize(imtest, norm= 'l2')
    return data, imtest



def train(num_patches, image, n_samples, w, h):
    for i in range(1, num_patches):
        imtrain = shuffle(image)
        imtrain = imtrain[:1000]

    '''
    Fit gaussian mixture model using 7 components repeatedly 
    with small random samples from the data using Expectation Maximization.
    '''
    print("Fitting Gaussain Mixture Model with Expectation Maximization")

    gmm = mixture.GaussianMixture(n_components=7, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=1200, n_init=1, init_params='kmeans', warm_start=True).fit(imtrain)

    # '''
    # Fit Bayesian gaussian mixture model using 7 components repeatedly 
    # with small random samples from the data using Dirichilet process.
    # '''
    # print("Fitting Bayesian Gaussian mixture with Dirichlet process")

    # dpgmm = mixture.BayesianGaussianMixture(n_components=7, covariance_type='full', weight_concentration_prior_type='dirichlet_distribution', tol=0.001, reg_covar=1e-06, max_iter=1200, n_init=1, init_params='kmeans', warm_start=True).fit(imtrain)

    return gmm



def test(im_test, gmm):
    labels = gmm.predict(im_test)
    return labels


def segment(image, samples, label, num_comp, w, h):
    label = np.expand_dims(label, axis=0)
    label = np.transpose(label)

    for i in range(1, num_comp):
        indices = np.where(np.all(label == i, axis=-1))
        indices = np.unravel_index(indices, (w, h), order='C')
        indices = np.transpose(indices)

        l =  chain.from_iterable(zip(*indices))

        for j, (lowercase, uppercase) in enumerate(l):
            image[lowercase, uppercase] = _color_codes[(i)]

    return image


if __name__ == "__main__":

    pass