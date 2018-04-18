#GMM class 
#This class initialized and stores the paramters of the Gaussian mixture model, 
#as well as storing the current noisy image being processed. The final 
#optimization is handled in mapEstimator.py

class GMM:
	#the key variables are the cropped image (window) and the GMM parameters
	__init__(self, window):
		#initialize variables (window, means, variances, etc)
		#perform initial estimation of GMM paramters (see Xu equations 15-18)

