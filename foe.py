#Field of Experts class
#This class stores the Field of Expert paramters, and also contains a method for training 
#them using noisy and ground truth images. This class is eventually used with the GMM class
#in mapEstimator.py 
class FoE:
	def __init__():
		#initialize variables (basis filters, alpha, beta)

	def load(filename):
		#load model from file

	def save(filename):
		#save model to file

	def train(noisyImage, trueImage):
		#compute loss gradient (see Xu equations 7-8)

		#update alpha and beta weights according to noisy and true images

		#possibly iterate until convergence

	def getPrior():
		#return prior represented by model
