Ryan Van Soelen
Project Description

This project aims to build a model that can denoise MRI images trained though supervised learning.
There are two components to this model: the prior distribution and the likelihood function. 
The prior is modeled using a variant of a higher-order Markov Random Field call a Field of Experts. 
The model is trained offline on a set of noisy and ground truth examples, optimized using the Newton method.
The likelihood function is a Gaussian Mixture Model, with parameters estimated using expectation maximization. 
Both the Field of Experts prior and the GMM likelihood function are combined to form an estimate of the posterior 
probability of the MRI images. This probability is maximized using MAP estimation and a quasi-Newton method called
L-BFGS. This optimization is done online and is used to estimate the most likely de-noised image based on the noisy 
image. It is not used during training, but during evaluation. Images are denoised using a sliding window, rather 
than all at once. "Denoising model for parallel magnetic resonance imaging images using higher-order Markov random 
fields" by Zhihuo Xu and Quan Shi is the primary source of the update equations for the two optimization steps. 

The project is broken up into multiple files:
train.py learns the Field of Experts model from the training data and saves the model. 
foe.py contains the Field of Experts class. 
gmm.py contains the Gaussian Mixture Model class.
mapEstimator.py contains the MAP estimator calss, which uses the L-BFGS method
denoise.py is the main file in the project and is used for evaluating trained models
util.py contains any global parameters or helper functions that may be needed.
Python libraries numpy and pillow is used for the matrix calcualtions and image manipulations.
Python library scipy is used for creating the convolutional filters and for optimization
Python library pickle will be used for saving and loading previously trained models. 


When submitting the pesudocode assignment, it was estimated that about 80% of the code would be completed by this 
time, not including debugging. The progress of the project has largely adhered to this expectation, and is beyond 
50% completion. All major classes and fucntions have most or all of thier code written. The most complex parts of 
the project (training and optimizing the FoE, GMM, and MAP models) have all of their code written. Furthermore, the 
general training and evaluation algorithms are nearly complete, with just the evaluation post-processing step 
missing. In the following weeks, efforts will be focused on finishing the remaining sections of code, and testing/
debugging the code. This is estimated to take about 1.5 weeks. In the remaining time, the model will be trained, and 
the final report will be written.


