#MAP Estimator class
#This class combines the GMM and the FoE models, and uses the L-BFGS method to 
#estimate the most likely denoised image given a noisy image (stored in the GMM class)
class MAPEstimator:
	def __init__(self, gmm, foe):
		#initialize variables (gmm, foe, etc)
		self.gmm = gmm
		self.foe = feo

	def estimate(self):
		#Use L-BFGS method to optimize GMM 
		#compute energy gradient as described in papers
			#get first term from foe model

			#get second term from GMM model

		#update prediction of true image according to gradient 
		#scipy.optimize.fmin_l_bfgs_b (limited memory, bounded) or scipy.optimize.fmin_bfgs (original)

		#repeat until convergence

		#output best prediction of true image