#MAP Estimator class
#This class combines the GMM and the FoE models, and uses the L-BFGS method to 
#estimate the most likely denoised image given a noisy image (stored in the GMM class)
from scipy import optimize
import numpy as np
import pdb

class MAPEstimator:
	def __init__(self, foe, gmm=None):
		#initialize variables (gmm, foe, etc)
		self.gmm = gmm
		self.foe = foe

	#Uses MAP estimation to estimate the most likely noise-free image
	#MAP is performed by minimizing the energy (log of proability)
	#Uses the L-BFGS method to optimize 
	def estimate(self):
		assert(self.gmm != None)
		#Use L-BFGS method to optimize GMM 
		#compute energy gradient as described in papers
		#get foe variables
		filters = self.foe.filters
		phi = self.foe.phi
		phiPrime = self.foe.phiPrime

		#get gmm variables
		z = self.gmm.z
		noisySegment = self.gmm.segment
		mu = self.gmm.mu
		sigma2 = self.gmm.sigma2
		pi = self.gmm.pi
		gaussian = self.gmm.gaussian
		weights = np.ones((1, self.gmm.numGaussians)) /self.gmm.segmentSize
		
		#Create functions for energy minimization
		def E(denoisedSegment): 
			denoisedSegment = denoisedSegment.reshape((-1, 1))
			ret = (np.sum(phi(np.matmul(filters, denoisedSegment))) 
							+ np.sum(weights*z
								*(np.log(pi)+np.log(gaussian(noisySegment-denoisedSegment, mu, sigma2))))
							)
			return ret

		def dE(denoisedSegment): 
			denoisedSegment = denoisedSegment.reshape((-1, 1))
			ret = (np.sum(np.matmul(filters.transpose((0, 2, 1)), phiPrime(np.matmul(filters, denoisedSegment))), axis=0)
							+ np.sum(weights*z
								*(noisySegment - denoisedSegment - mu)/sigma2, axis=1, keepdims=True)
					).reshape((-1))
			return ret

		#Use L-BFGS to estimate the most probable image
		guessImage = noisySegment
		denoised, value, info = optimize.fmin_l_bfgs_b(E, guessImage, fprime=dE)
		#output best prediction of true image
		return denoised