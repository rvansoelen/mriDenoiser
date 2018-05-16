#GMM class 
#This class initialized and stores the paramters of the Gaussian mixture model, 
#as well as storing the current noisy image being processed. The final 
#optimization is handled in mapEstimator.py
import numpy as np
import pdb

class GMM:
	#the key variables are the cropped image (segment) and the GMM parameters
	def __init__(self, segment):
		#flatten segment image 
		self.segment = segment.reshape((-1, 1))
		self.segmentSize = self.segment.size
		self.numGaussians = 5
		self.mu = np.zeros((1, self.numGaussians)) #might have to use random intial values
		self.sigma2 = np.ones((1, self.numGaussians)) #might have to use random intial values
		self.pi = np.ones((1, self.numGaussians))/float(self.numGaussians) #might have to use random intial values
		self.z = self.calculateZ(self.mu, self.sigma2, self.pi)
		self.maxIterations = 80
		self.errorTolerance = 0.001

		#perform initial estimation of GMM paramters (see Xu equations 15-18)
		self.expectationMaximisation()

	#Estimates the GMM parameters based on the noisy image segment
	#Updates the parameters according to Expectation Maximization, see paper for details
	def expectationMaximisation(self):
		#update weights until convergence	
		convergence = False
		numIterations = 0
		while True:
			#Temporarily store previous iteration's parameters
			oldMu = self.mu
			oldSigma2 = self.sigma2
			oldZ = self.z
			oldZT = oldZ.transpose((1, 0))
			oldPi = self.pi
			segmentT = self.segment.transpose((1, 0))

			#update the parameters 
			self.pi = np.sum(oldZ, axis=0, keepdims=True)/float(self.segmentSize)
			self.mu = np.matmul(segmentT, oldZ)/np.sum(oldZ , axis=0, keepdims=True)
			self.sigma2 = np.sum(oldZ*(self.segment-oldMu)**2, axis=0, keepdims=True)/np.sum(oldZ , axis=0, keepdims=True)
			self.z = self.calculateZ(oldMu, oldSigma2, oldPi)

			#compute the difference between the currrent and previous iterations
			diffMu = np.sum(np.abs(self.mu-oldMu))
			diffSigma2 = np.sum(np.abs(self.sigma2-oldSigma2))
			diffZ = np.sum(np.abs(self.z-oldZ))
			diffPi = np.sum(np.abs(self.pi-oldPi))
			err = diffMu + diffSigma2 + diffZ + diffPi

			#Break if error tolerance is met or if max number of iterations is met
			if err < self.errorTolerance:
				break
			elif numIterations > self.maxIterations:
				print('Warning: Maximum number of iterations reached during Gaussian learning')
				break
			numIterations += 1

	#Calculates the variable z from the given parameters, see paper for details
	def calculateZ(self, mu, sigma2, pi):
		numerator = pi*self.gaussian(self.segment, mu, sigma2)
		return numerator/np.sum(numerator, axis=1, keepdims=True)

	#Calculates the Gaussian distrbution of the given parameters, evaluated at the given image
	def gaussian(self, image, mu, sigma2):
		return np.exp(-(image - mu)**2/(2*sigma2))/np.sqrt(2*np.pi*sigma2)