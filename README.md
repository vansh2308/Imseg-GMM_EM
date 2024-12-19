# GMM-EM for Image Segmentation 
___
#### Image Segmentation using Gaussian Mixture Models with Expectation Maximization on colour and texture descriptors

<br/>
<img src="/input/Group 1.png" width="100%">
<br/><br/>

Gaussian Filter is applied on the image and projected to the CIELAB space. Local Texture information is caputred using Local Binary Patterns. Next, the image is flattened and normalized.

The approach models the image data distribution as a Gaussian Mixture Model. By randomly sampling 1000 data points (pixels from the image) over 1000 iterations, we could estimate the parameters (means/labels, covariances and joint membership weights of a point belonging to each cluster) by repeated adjustments with Expectation Maximization Agorithm to derive optimal cluster and labels.

ToDos:
- [ ] Hyperparamter Tuning
- [ ] Plotting the Gaussian Mixture distribution

#### Setup Instructions
```sh
git clone https://github.com/vansh2308/Imseg-GMM_EM.git
cd ./Imseg-GMM_EM

chmod +x run.sh
./run.sh
```

Voila 🙂 You might tweak the input file in `run.sh` and experiment!
<br/>

#### Author
- Github - [vansh2308](https://github.com/vansh2308)
- Website - [Vansh Agarwal](https://portfolio-website-self-xi.vercel.app/)

