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
		self.numFilters = 48
		self.numBasisFilters = self.filterSize * self.filterSize
		self.alpha = np.random.rand(self.numFilters, 1, 1) #random values from [0, 1)
		#One 
		self.beta = np.random.rand(self.numFilters, self.filterSize, self.filterSize)
		self.alphaStepSize = 0.001
		self.betaStepSize = 0.001
		self.windowSizeX = windowSizeX
		self.windowSizeY = windowSizeY
		assert(self.windowSizeX > self.filterSize and self.windowSizeY > self.filterSize)
		self.basisFilters = self.computeConvBasisFilters()
		self.filters = self.computeConvFilters()
		

	def load(filename):
		#load model from file
		with open(filename, 'rb') as fid:
			foe = pickle.load(fid)
			return foe

	def save(self, filename):
		#save model to file
		with open(filename,'wb') as fid:
			pickle.dump(self,fid) 

	#Computes the convolutional filters as matrix operators
	def computeConvFilters(self):
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
		return convMatrices

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
		convMatrices = np.zeros((self.numBasisFilters, (n - k + 1) * (m - k + 1), n*m))
		#flatenedFilters = np.zeros((self.numFilters, self.filterSize, self.filterSize))
		#flatenedFilters[:, :self.filterSize, :self.filterSize] = self.filters
		#flatenedFilters = np.reshape(flatenedFilters, (self.numFilters, 1, -1))
		flatenedFilters = filters2D.reshape((self.numBasisFilters, -1))
		flattenedFilterLength = self.filterSize*self.filterSize

		for i in range((n - k + 1) * (m - k + 1) - flattenedFilterLength +1):
			convMatrices[:, i, i:i+flattenedFilterLength] = flatenedFilters
		return convMatrices

	'''
	#The derivative of the energy term, see paper for details 
	#not sure if correct
	def deltaE(self, x, noisyFlatImage):
		filters = self.computeConvFilters()
		sum = 0
		conv1 = np.dot(filters, x)
		phi_prime = self.phiPrime(conv1)
		conv2 = self.alpha*np.dot(filters.transpose((0, 2, 1)), phi_prime)
		deltaE = np.sum(conv2, axis=0) + x - noisyFlatImage
		return deltaE
	'''

	#The energy term, see paper for details 
	def E(self, estimatedImage, trueImage):
		filters = self.filters
		conv = filters.dot(estimatedImage)
		print 'Estimated Image', estimatedImage
		print 'Conv: ', conv
		energy = (np.sum(self.phi(conv)) 
			+ 0.5*np.sum((trueImage-estimatedImage)**2))
		print energy
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
		phiPP = self.phiPrimePrime(np.dot(filters, x))
		print phiPP.shape
		d = np.zeros((self.numFilters, phiPP.shape[1], phiPP.shape[1]))
		for n in range(self.numFilters):
			print 'diag'
			d[n, :, :] = np.diagflat(phiPP[n, :])
		return d

	#The hessian matrix required for optimization, see paper for details
	def hessian(self, x):
		filters = self.computeConvFilters()
		d = self.diagonal(x)
		print 'uuio1'
		conv1 = np.dot(filters.transpose((0, 2, 1)), d)
		print 'uuio2'
		conv2 = np.dot(conv1, filters)
		print 'uuio3'
		return sum(self.alpha*conv2, axis=0) + np.identity(conv2.shape[1])

	#The main function for training the FoE model, one sample at a time 
	def train(self, noisyImageBatch, trueImageBatch):
		#compute gradients for batch, only updating at the end
		deltaAlpha = np.zeros((self.numFilters))
		deltaBeta = np.zeros((self.numFilters, self.numBasisFilters))
		print len(noisyImageBatch)
		print len(trueImageBatch)
		for noisyImage, trueImage in itertools.izip(noisyImageBatch, trueImageBatch):
			print 'i'
			noisyFlat = noisyImage.reshape(-1, 1)
			trueFlat = trueImage.reshape(-1, 1)
			#flatten images 
			#noisyFlat = np.reshape(noisyImageBatch[index, :, :], (-1, 1))
			#trueFlat = np.reshape(trueImage[index, :, :], (-1, 1))
			#estimate image 
			guess = np.random.rand(self.windowSizeY*self.windowSizeX, 1)
			#print self.deltaE(guess, noisyFlat).shape
			print guess.shape
			#estimate = optimize.newton(lambda x : self.deltaE(x, noisyFlat), 0)#guess)
			#temporarily get rid of:
			'''
			result = optimize.minimize(self.E, guess, args=(noisyFlat), options={'maxiter':5, 'disp':True})
			estimate = result.x
			if not result.success:
				print 'Low Level FoE optimizer exited without success'
			else:
				print 'Success'
			'''
			estimate = guess
			#compute loss gradient (see Xu equations 7-8)
			#compute diagonal D
			d = self.diagonal(estimate)
			#compute Hessian 
			h = self.hessian(estimate)
			print 'inverting hessian'
			hInv = np.linalg.tensorinv(h)
			print 'inverted hessian'
			#compute basis filters 
			b = self.basisFilters
			bt = b.transpose((0, 2, 1))
			#compute learned filters
			f = self.filters
			ft = f.transpose((0, 2, 1))

			print 'Updating weights'
			#update alpha and beta weights according to noisy and true images
			deltaAlpha += -(ft.dot(phiPrime(np.dot(f, guess)))).transpose((0, 2, 1)).dot(hInv).dot(guess-trueImage)

			deltaBeta += -(bt.dot(phiPrime(np.dot(f, guess))) + ft.dot(d).dot(b).dot(guess)).transpose((0, 2, 1)).dot(hInv).dot(guess-trueImage)
			print 'Updated weights'
		#update weights and filters
		deltaBeta = deltaBeta.reshape((self.numFilters, self.filterSize, self.filterSize))
		self.alpha = self.alpha - self.alphaStepSize*deltaAlpha
		self.beta = self.beta - self.betaStepSize*deltaBeta
		self.filters = self.computeConvFilters()



