#MAP Estimator class
#This class combines the GMM and the FoE models, and uses the L-BFGS method to 
#estimate the most likely denoised image given a noisy image (stored in the GMM class)
from scipy import optimize

class MAPEstimator:
	def __init__(self, gmm, foe):
		#initialize variables (gmm, foe, etc)
		self.gmm = gmm
		self.foe = feo

	def estimate(self):
		#Use L-BFGS method to optimize GMM 
		#compute energy gradient as described in papers
			#get foe variables
			alpha = foe.alpha
			filters = foe.filters
			phi = foe.phi
			phiPrime = foe.phiPrime

			#get gmm variables
			z = gmm.z
			noisyWindow = gmm.window
			mu = gmm.mu
			sigma2 = gmm.sigma2
			pi = gmm.pi
			weights = np.ones(gmm.windowSize)/gmm.windowSize

			E = lambda denoisedWindow: (np.sum(alpha*phi(filters.dot(denoisedWindow)), axis=0) 
								+ np.sum(weights*z
									*(np.log(pi)+np.log(gaussian(denoisedWindow, mu, sigma, pi))), axis=1)
								)

			dE = lambda denoisedWindow: (np.sum(alpha*phiPrime(filters.transpose((0, 2, 1)).dot(denoisedWindow)), axis=0)
								+ np.sum(weights*z
									*(noisyWindow - denoisedWindow - mu)/sigma2, axis=1)
								)

		#Use L-BFGS to estimate the most probable image
		#scipy.optimize.fmin_l_bfgs_b (limited memory, bounded) or scipy.optimize.fmin_bfgs (original)
		denoised, value, info = optimize.fmin_l_bfgs_b(E, guessImage, fprime=dE)

		#output best prediction of true image
		return denoised