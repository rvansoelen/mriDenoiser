Ryan Van Soelen
Project Description

This project aims to build a model that can denoise MRI images trained though supervised learning.
There are two components to this model: the prior distribution and the likelihood function. 
The prior is modeled using a variant of a higher-order Markov Random Field call a Field of Experts. 
The model is trained on a set of noisy and ground truth examples, optimized using the Newton method.
The likelihood function is a Gaussian Mixture Model. Both the Field of Experts prior and the GMM 
likelihood function are combined to form an estimate of the posterior probability of the MRI images.
This is optimized using a MAP estimator and a quasi-Newton method called L-BFGS. This optimization is done online and is used to estimate the most likely de-noised image based on the noisy image. It is not used during training, but during evaluation. Images are denoised using a sliding window, rather than all at once. "Denoising model for parallel magnetic resonance imaging images using higher-order Markov random fields" by Zhihuo Xu and Quan Shi is the primary source of the update equations for the two optimization steps. 

The project is broken up into multiple files:
train.py learns the Field of Experts model from the training data and saves the model. 
foe.py contains the Field of Experts class. 
gmm.py contains the Gaussian Mixture Model class.
mapEstimator.py contains the MAP estimator calss, which uses the L-BFGS method
denoise.py is the main file in the project and is used for evaluating trained models
util.py contains any global parameters or helper functions that may be needed.
Python libraries numpy and pillow with be used for the matrix calcualtions and image manipulations.

Since the training and evaluation steps are fairly straight-forward, it is estimated that about
80%-100% of the code can be written by the next assignment deadline, on 4/24. However, the code 
will most-likely be buggy, so more work would have to be done to test and clean the code afterwards.
