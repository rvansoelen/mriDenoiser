#GMM class 
#This class initialized and stores the paramters of the Gaussian mixture model, 
#as well as storing the current noisy image being processed. The final 
#optimization is handled in mapEstimator.py

import math

class GMM:
	#the key variables are the cropped image (window) and the GMM parameters
	def __init__(self, window):
		#flatten window image 
		self.window = window.reshape((1, -1))
		self.windowSize = self.window.size()
		self.numGaussians = 5
		self.mu = np.zeros(self.numGaussians, 1)
		self.sigma2 = np.ones(self.numGaussians, 1)
		self.pi = np.ones(self.windowSize, 1)/self.windowSize
		self.z = self.getZ(self.mu, self.sigma2, self.pi)
		self.maxIterations = 10000
		self.errorTolerance = 0.001

		#perform initial estimation of GMM paramters (see Xu equations 15-18)
		self.expectationMaximisation()

	def expectationMaximisation(self):
		#update weights until convergence	
		convergence = false
		numIterations = 0
		while True:
			oldMu = self.mu
			oldSigma2 = self.sigma2
			oldZ = self.z
			oldPi = self.pi

			self.pi = np.sum(oldZ, axis=1)/self.windowSize
			self.mu = np.sum(oldZ*self.window, axis=0)/np.sum(oldZ , axis=0)
			self.sigma2 = np.sum(oldZ*(self.window-oldMu)**2)/np.sum(oldZ , axis=0)
			self.z = self.calculateZ(oldMu, oldSigma2, oldPi)

			diffMu = np.sum(np.abs(self.mu-oldMu))
			diffSigma2 = np.sum(np.abs(self.sigma2-oldSigma2))
			diffZ = np.sum(np.abs(self.z-oldZ))
			diffPi = np.sum(np.abs(self.pi-oldPi))

			if diffMu + diffSigma2 + diffZ + diffPi < self.errorTolerance:
				break
			else if numIterations < self.maxIterations:
				print('Warning: Maximum number of iterations reached during Gaussian learning')
				break

	def calculateZ(self, mu, sigma2, pi):
		numerator = pi*self.gaussian(self.window, mu, sigma2)
		return numerator/np.sum(numerator, axis=1)

	def gaussian(self, image, mu, sigma2):
		return np.exp(-(image - mu)**2/(2*sigma2))/np.sqrt(2*math.pi*sigma2)