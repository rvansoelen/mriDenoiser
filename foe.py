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

class FoE:
	def __init__(self, windowSizeX=util.windowWidth, windowSizeY=util.windowHeight):
		#initialize variables (basis filters, alpha, beta)
		self.filterSize = 7
		self.numFilters = 5
		self.numBasisFilters = self.filterSize * self.filterSize
		self.alpha = np.random.rand(self.numFilters, 1, 1) #random values from [0, 1)
		#One 
		self.beta = np.random.rand(self.numFilters, self.numBasisFilters)
		self.alphaStepSize = 0.01
		self.betaStepSize = 0.01
		self.windowSizeX = windowSizeX
		self.windowSizeY = windowSizeY
		assert(self.windowSizeX > self.filterSize and self.windowSizeY > self.filterSize)
		self.basisFilters = self.computeConvBasisFilters()
		self.filters = self.computeConvFilters()
		self.errorHistory = []
		#self.brk = True

	def save(self, filename):
		#save model to file
		with open(filename,'wb') as fid:
			pickle.dump(self,fid) 

	#Computes the convolutional filters as matrix operators
	def computeConvFilters(self):
		'''
		filters2D = fftpack.idct(fftpack.idct(self.beta, axis=1), axis=2)
		#must flatten filters and convert to convolutional matrices
		n = self.windowSizeY
		m = self.windowSizeX
		k = self.filterSize
		assert(n>=k and m>=k)
		outputSize = (n-k+1)*(m-k+1)
		convMatrices = np.zeros((self.numFilters, outputSize, n*m))
		#create filters with spacing on x axis
		filters = np.zeros((self.numFilters, k, m))
		filters[:, :, :k] = filters2D
		#flatten filters and delete excess zeros
		flattenedFilterLength = k*m-(m-k)
		flattenedFilters = filters.reshape(self.numFilters, k*m)[:, :flattenedFilterLength]
		#assume convolution has no padding 
		#pdb.set_trace()
		for i in range(outputSize):
			convMatrices[:, i, i:i+flattenedFilterLength] = flattenedFilters
		print np.sum(np.abs(convMatrices))
		'''
		return np.tensordot(self.beta, self.basisFilters, axes=([1, 0]))

	#Computes the basis filters that when combined form the convolutional filters (structured as matrix operators)
	#NOT CORRECT
	def computeConvBasisFilters(self):
		frequencies = np.zeros((self.numBasisFilters, self.filterSize, self.filterSize))
		for i in range(self.filterSize):
			for j in range(self.filterSize):
				frequencies[i*self.filterSize+j, i, j] = 1
		filters2D = fftpack.idct(fftpack.idct(frequencies, axis=1), axis=2)
		#must flatten filters and convert to convolutional matrices
		n = self.windowSizeY
		m = self.windowSizeX
		k = self.filterSize
		outputSize = (n-k+1)*(m-k+1)
		convMatrices = np.zeros((self.numBasisFilters, outputSize, n, m))
		#assume convolution has no padding 
		for i in range(outputSize):
			convMatrices[:, i, i/(m-k+1):i/(m-k+1)+k, i%(m-k+1):i%(m-k+1)+k] = filters2D
		#flatten so that images are inputed are flattend arrays
		return convMatrices.reshape((self.numBasisFilters, outputSize, n*m))

	
	#The derivative of the energy term, see paper for details 
	def jacE(self, x, noisyFlatImage):
		x = x.reshape((-1, 1))
		noisyFlatImage = noisyFlatImage.reshape((-1, 1))
		filters = self.filters
		conv1 = np.matmul(filters, x)
		phi_prime = self.phiPrime(conv1)
		conv2 = self.alpha*np.matmul(filters.transpose((0, 2, 1)), phi_prime)
		deltaE = np.sum(conv2, axis=0) + x - noisyFlatImage
		#print 'Jac', deltaE.reshape((-1))
		return deltaE.reshape((-1))
	

	#The energy term, see paper for details 
	def E(self, estimatedImage, noisyFlatImage):
		estimatedImage = estimatedImage.reshape((-1, 1))
		noisyFlatImage = noisyFlatImage.reshape((-1, 1))
		filters = self.filters
		conv = np.matmul(filters, estimatedImage)
		#print 'Estimated Image', estimatedImage
		#print 'Conv: ', conv
		energy = (np.sum(self.phi(conv)) 
			+ 0.5*np.sum((noisyFlatImage-estimatedImage)**2))
		#print energy
		#if self.brk:
		#	pdb.set_trace()
		return energy

	#A function used in the FoE model, see paper for details
	def phi(self, x):
		return self.alpha*np.log(1+x**2)

	#The first derivative of the phi function 
	def phiPrime(self, input):
		return self.alpha*2*input/(1+input**2)

	#The second derivative of the phi function
	def phiPrimePrime(self, input):
		return self.alpha*2*(1-input**2)/(1+input**2)**2

	#The diagonal matrix required for optimization, see paper for details
	def diagonal(self, x):
		filters = self.filters
		phiPP = self.phiPrimePrime(np.matmul(filters, x))
		#print phiPP.shape
		d = np.zeros((self.numFilters, phiPP.shape[1], phiPP.shape[1]))
		for n in range(self.numFilters):
			#print 'diag'
			d[n, :, :] = np.diagflat(phiPP[n, :])
		return d

	#The hessian matrix required for optimization, see paper for details
	def hessian(self, x, placeholder=None): #placeholder is required for scipy.optimize.minimize
		filters = self.filters
		x = x.reshape((-1, 1))
		d = self.diagonal(x)
		#print 'uuio1'
		#print 'filters transpose: ', filters.transpose((0, 2, 1)).shape
		#print 'd:', d.shape
		#print np.ascontiguousarray(filters.transpose((0, 2, 1))).flags
		#print d.flags
		#pdb.set_trace()
		#Must make arrays C_CONTIGUOUS!!!
		conv1 = np.matmul(filters.transpose((0, 2, 1)), d)
		#print 'uuio2'
		conv2 = np.matmul(conv1, filters)
		#print 'uuio3'
		#pdb.set_trace()
		return np.sum(self.alpha*conv2, axis=0) + np.identity(conv2.shape[1])

	#The main function for training the FoE model, one sample at a time 
	def train(self, segmentPairBatch):
		#compute gradients for batch, only updating at the end
		deltaAlpha = np.zeros(self.alpha.shape)
		deltaBeta = np.zeros(self.beta.shape)
		#i=0
		self.errorHistory = []
		for noisyImage, trueImage in segmentPairBatch:
			#print 'Iteration: ', i
			#i += 1
			#if i > 10: break
			noisyFlat = noisyImage.reshape(-1, 1)
			trueFlat = trueImage.reshape(-1, 1)
			#flatten images 
			#noisyFlat = np.reshape(noisyImageBatch[index, :, :], (-1, 1))
			#trueFlat = np.reshape(trueImage[index, :, :], (-1, 1))
			#estimate image 
			guess = noisyImage.reshape((-1)) #np.random.rand(self.windowSizeY*self.windowSizeX, 1)
			#print self.deltaE(guess, noisyFlat).shape
			#print guess.shape
			#estimate = optimize.newton(lambda x : self.deltaE(x, noisyFlat), 0)#guess)
			#temporarily get rid of:
			
			result = optimize.minimize(self.E, guess, method='Newton-CG', jac=self.jacE, hess=self.hessian, args=(noisyFlat)) #, options={'disp':True})#'maxiter':500,
			#pdb.set_trace()
			estimate = result.x.reshape((-1, 1))
			'''if not result.success:
				print 'Low Level FoE optimizer exited without success'
			else:
				print 'Success'
			'''
			#estimate = guess
			#compute loss gradient (see Xu equations 7-8)
			#compute diagonal D
			d = self.diagonal(estimate)
			#compute Hessian 
			h = self.hessian(estimate)
			#print 'inverting hessian'
			#hInv = np.linalg.tensorinv(h)
			hInv = np.linalg.inv(h)
			#print 'inverted hessian'
			#compute basis filters 
			b = self.basisFilters
			bt = b.transpose((0, 2, 1))
			#compute learned filters
			f = self.filters
			ft = f.transpose((0, 2, 1))
			#pdb.set_trace()
			#print 'Updating weights'
			#update alpha and beta weights according to noisy and true images

			#print test.shape
			#print (guess-trueFlat).shape
			#print deltaAlpha.shape
			#print self.alpha.shape
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
							
			#print ''
			#print f.shape
			#print d.shape
			#print b.shape
			#print guess.shape
			#print deltaBeta.shape
			#print self.beta.shape
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
		self.alpha = self.alpha - self.alphaStepSize*deltaAlpha
		self.beta = self.beta - self.betaStepSize*deltaBeta
		self.filters = self.computeConvFilters()
		err = np.mean((estimate-trueFlat)**2)
		print 'L2 Error: ', err
		self.errorHistory.append(err)
		#print "Finished Batch"


def load(filename):
		#load model from file
		with open(filename, 'rb') as fid:
			foe = pickle.load(fid)
		return foe