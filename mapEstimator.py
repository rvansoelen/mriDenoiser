#MAP Estimator class
#This class combines the GMM and the FoE models, and uses the L-BFGS method to 
#estimate the most likely denoised image given a noisy image (stored in the GMM class)
class MAPEstimator:
	def __init__(self, gmm, foe):
		#initialize variables (gmm, foe, etc)

	def estimate(self):
		#Use L-BFGS method to optimize GMM 
			#compute energy gradient as described in papers

			#update prediction of true image according to gradient (see Xu equations 20-21)

			#repeat until convergence

		#output best prediction of true image