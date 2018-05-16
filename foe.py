#Field of Experts class
#This class stores the Field of Expert paramters, and also contains a method for training 
#them using noisy and ground truth images. This class is eventually used with the GMM class
#in mapEstimator.py 
import numpy as np
from scipy import signal, fftpack, optimize
import pickle
import util
import itertools
import pdb
import time


class FoE:
	def __init__(self, segmentSizeX=util.segmentWidth, segmentSizeY=util.segmentHeight):
		#initialize variables (basis filters, alpha, beta, etc.)
		self.filterSize = 6 #Assume a square filter
		self.numFilters = 5
		self.numBasisFilters = self.filterSize * self.filterSize
		self.alpha = np.random.rand(self.numFilters, 1, 1)/self.numFilters #random values from [0, 1)
		self.beta = np.random.rand(self.numFilters, self.numBasisFilters)/(self.numFilters*self.numBasisFilters)
		self.alphaStepSize = 0.01
		self.betaStepSize = 0.01
		self.segmentSizeX = segmentSizeX
		self.segmentSizeY = segmentSizeY
		#Segment dimensions must be greater than the filter size
		assert(self.segmentSizeX > self.filterSize and self.segmentSizeY > self.filterSize)
		self.basisFilters = self.computeConvBasisFilters()
		self.filters = self.computeConvFilters()
		self.errorHistory = []

	#Saves this object to a file
	def save(self, filename):
		#save model to file
		with open(filename,'wb') as fid:
			pickle.dump(self,fid) 

	#Computes the convolutional filters as matrix operators
	def computeConvFilters(self):
		return np.tensordot(self.beta, self.basisFilters, axes=([1, 0]))

	#Computes the basis filters that when combined 
	#form the convolutional filters (structured as matrix operators)
	def computeConvBasisFilters(self):
		frequencies = np.zeros((self.numBasisFilters, self.filterSize, self.filterSize))
		for i in range(self.filterSize):
			for j in range(self.filterSize):
				frequencies[i*self.filterSize+j, i, j] = 1
		filters2D = fftpack.idct(fftpack.idct(frequencies, axis=1), axis=2)
		#must flatten filters and convert to convolutional matrices
		n = self.segmentSizeY
		m = self.segmentSizeX
		k = self.filterSize
		outputSize = (n-k+1)*(m-k+1)
		convMatrices = np.zeros((self.numBasisFilters, outputSize, n, m))
		#assume convolution has no padding 
		for i in range(outputSize):
			convMatrices[:, i, i/(m-k+1):i/(m-k+1)+k, i%(m-k+1):i%(m-k+1)+k] = filters2D
		#flatten so that images are inputed are flattend arrays
		return convMatrices.reshape((self.numBasisFilters, outputSize, n*m))

	
	#The derivative (jaccobian) of the energy term, see paper for details 
	def jacE(self, estimatedFlatImage, noisyFlatImage):
		#reshape segment vectors to make sure the math broadcasts correctly
		estimatedFlatImage = estimatedFlatImage.reshape((-1, 1))
		noisyFlatImage = noisyFlatImage.reshape((-1, 1))
		#Compute the jaccobian
		filters = self.filters
		conv1 = np.matmul(filters, estimatedFlatImage)
		phi_prime = self.phiPrime(conv1)
		conv2 = np.matmul(filters.transpose((0, 2, 1)), phi_prime)
		deltaE = np.sum(conv2, axis=0) + estimatedFlatImage - noisyFlatImage
		return deltaE.reshape((-1)) #re-flatten to 1D vector
	

	#The energy term, see paper for details 
	def E(self, estimatedImage, noisyFlatImage):
		#reshape segment vectors to make sure the math broadcasts correctly
		estimatedImage = estimatedImage.reshape((-1, 1))
		noisyFlatImage = noisyFlatImage.reshape((-1, 1))
		#Calculate the energy
		conv = np.matmul(self.filters, estimatedImage)
		energy = (np.sum(self.phi(conv)) 
			+ 0.5*np.sum((noisyFlatImage-estimatedImage)**2))
		return energy

	#A function used in the FoE model, see paper for details
	def phi(self, input):
		return self.alpha*np.log(1+0.5*input**2)

	#The first derivative of the phi function 
	def phiPrime(self, input):
		return self.alpha*input/(1+0.5*input**2)

	#The second derivative of the phi function
	def phiPrimePrime(self, input):
		return self.alpha*(1-0.5*input**2)/(1+0.5*input**2)**2

	#The diagonal matrix required for optimization, see paper for details
	def diagonal(self, input):
		filters = self.filters
		phiPP = self.phiPrimePrime(np.matmul(filters, input))
		d = np.zeros((self.numFilters, phiPP.shape[1], phiPP.shape[1]))
		for n in range(self.numFilters):
			d[n, :, :] = np.diagflat(phiPP[n, :])
		return d

	#The hessian matrix required for optimization, see paper for details
	def hessian(self, estimatedFlatImage, placeholder=None): #placeholder is required for scipy.optimize.minimize
		filters = self.filters
		estimatedFlatImage = estimatedFlatImage.reshape((-1, 1))
		d = self.diagonal(estimatedFlatImage)
		conv1 = np.matmul(filters.transpose((0, 2, 1)), d)
		conv2 = np.matmul(conv1, filters)
		return np.sum(conv2, axis=0) + np.identity(conv2.shape[1])

	#The main function for training the FoE model, one batch at a time 
	#Training involves a bi-level optimization process
	#One call to train() updates the weights of the top-level optimization once
	def train(self, segmentPairBatch):
		#compute gradients for batch, only updating at the end
		start = time.time()
		deltaAlpha = np.zeros(self.alpha.shape)
		deltaBeta = np.zeros(self.beta.shape)
		batchErrorHistory = []
		for noisyImage, trueImage in segmentPairBatch:
			noisyFlat = noisyImage.reshape(-1, 1)
			trueFlat = trueImage.reshape(-1, 1)
			guess = noisyImage.reshape((-1)) 
			
			#Perform low-level optimization of energy
			result = optimize.minimize(self.E, guess, method='Newton-CG', jac=self.jacE, hess=self.hessian, args=(noisyFlat), options={'xtol': 1e-03, 'eps': 1e-04, 'maxiter': 150,'disp':False})
			estimate = result.x.reshape((-1, 1))
			if not result.success:
				print 'Low Level FoE optimizer exited without success'
			
			err = np.sum((estimate-trueFlat)**2)
			batchErrorHistory.append(err)
			
			#Perform top-level optimization of loss
			#compute loss gradient (see Xu equations 7-8)
			#compute diagonal D
			d = self.diagonal(estimate)
			#compute Hessian 
			h = self.hessian(estimate)
			hInv = np.linalg.inv(h)
			#compute basis filters 
			b = self.basisFilters
			bt = b.transpose((0, 2, 1))
			#compute learned filters
			f = self.filters
			ft = f.transpose((0, 2, 1))
			#update alpha and beta weights according to noisy and true images

			deltaAlpha += 	-np.matmul(
								np.matmul(
									np.matmul(
										ft, 
										self.phiPrime(np.matmul(
												f, 
												estimate
											)
										)
									).transpose((0, 2, 1)),
	 								hInv
								), 
								estimate-trueFlat
							)
			deltaBeta += 	-np.matmul(
								np.matmul(
									(np.dot(
										bt, 
										self.phiPrime(
											np.matmul(
												f, 
												estimate
											)
										)
									).transpose((2, 0, 1, 3))
									+ 
									np.matmul(
										np.dot(
											np.matmul(
												ft, 
												d
											), 
											b
										).transpose((0, 2, 1, 3)), 
										estimate
									)).transpose((0, 1, 3, 2)),
									hInv
								), 
								estimate-trueFlat
							).reshape(self.beta.shape)
		#update weights and filters
		self.alpha = self.alpha + self.alphaStepSize*deltaAlpha
		self.beta = self.beta + self.betaStepSize*deltaBeta
		self.filters = self.computeConvFilters()
		self.errorHistory.extend(batchErrorHistory)
		print 'Mean L2 Error over batch: ', np.mean(batchErrorHistory)
		if np.any(self.alpha < 0):
			print 'Warning: Some elements of alpha are negative'
		print 'Batch time: ', time.time()-start

#loads a previous FoE model
def load(filename):
		#load model from file
		with open(filename, 'rb') as fid:
			foe = pickle.load(fid)
		return foe