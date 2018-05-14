#GMM class 
#This class initialized and stores the paramters of the Gaussian mixture model, 
#as well as storing the current noisy image being processed. The final 
#optimization is handled in mapEstimator.py
import numpy as np
import pdb

class GMM:
	#the key variables are the cropped image (window) and the GMM parameters
	def __init__(self, window):
		#flatten window image 
		self.window = window.reshape((-1, 1))
		self.windowSize = self.window.size
		self.numGaussians = 5
		self.mu = np.zeros((1, self.numGaussians)) #might have to use random intial values
		self.sigma2 = np.ones((1, self.numGaussians)) #might have to use random intial values
		self.pi = np.ones((1, self.numGaussians))/float(self.numGaussians) #might have to use random intial values
		self.z = self.calculateZ(self.mu, self.sigma2, self.pi)
		self.maxIterations = 1000
		self.errorTolerance = 0.001

		#perform initial estimation of GMM paramters (see Xu equations 15-18)
		self.expectationMaximisation()

	#Estimates the GMM parameters based on the noisy image window
	#Updates the parameters according to Expectation Maximization, see paper for details
	def expectationMaximisation(self):
		#update weights until convergence	
		convergence = False
		numIterations = 0
		while True:
			oldMu = self.mu
			oldSigma2 = self.sigma2
			oldZ = self.z
			oldZT = oldZ.transpose((1, 0))
			oldPi = self.pi
			windowT = self.window.transpose((1, 0))

			self.pi = np.sum(oldZ, axis=0, keepdims=True)/float(self.windowSize)
			self.mu = np.matmul(windowT, oldZ)/np.sum(oldZ , axis=0, keepdims=True)
			self.sigma2 = (oldZ*(self.window-oldMu)**2)/np.sum(oldZ , axis=0, keepdims=True)
			self.z = self.calculateZ(oldMu, oldSigma2, oldPi)

			diffMu = np.sum(np.abs(self.mu-oldMu))
			diffSigma2 = np.sum(np.abs(self.sigma2-oldSigma2))
			diffZ = np.sum(np.abs(self.z-oldZ))
			diffPi = np.sum(np.abs(self.pi-oldPi))

			if diffMu + diffSigma2 + diffZ + diffPi < self.errorTolerance:
				break
			elif numIterations < self.maxIterations:
				print('Warning: Maximum number of iterations reached during Gaussian learning')
				break

	#Calculates the variable z from the given parameters, see paper for details
	def calculateZ(self, mu, sigma2, pi):
		numerator = pi*self.gaussian(self.window, mu, sigma2)
		return numerator/np.sum(numerator, axis=1, keepdims=True)

	#Calculates the Gaussian distrbution of the given parameters, evaluated at the given image
	def gaussian(self, image, mu, sigma2):
		return np.exp(-(image - mu)**2/(2*sigma2))/np.sqrt(2*np.pi*sigma2)