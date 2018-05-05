#Field of Experts class
#This class stores the Field of Expert paramters, and also contains a method for training 
#them using noisy and ground truth images. This class is eventually used with the GMM class
#in mapEstimator.py 
import numpy as np
from scipy import signal, fftpack
import pickle
import util

class FoE:
	def __init__(self, windowSizeX=util.windowWidth, windowSizeY=util.windowHeight):
		#initialize variables (basis filters, alpha, beta)
		self.filterSize = 7
		self.numFilters = 48
		self.numBasisFilters = self.filterSize * self.filterSize
		self.alpha = np.random.rand(self.numFilters) #random values from [0, 1)
		#One 
		self.beta = np.random.rand(self.numFilters, self.filterSize, self.filterSize)
		self.alphaStepSize = 0.001
		self.betaStepSize = 0.001
		self.windowSizeX = windowSizeX
		self.windowSizeY = windowSizeY
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
		assert(self.windowSizeX > self.filterSize and self.windowSizeY > self.filterSize)
		n = self.windowSizeY
		m = self.windowSizeX
		convMatrices = zeros((self.numFilters, (n - k + 1) * (m - k + 1), n*m))
		flatenedFilters = np.zeros((self.numFilters, self.filterSize, self.filterSize))
		flatenedFilters[:, :self.filterSize, :self.filterSize] = self.filters
		flatenedFilters = np.reshape(flatenedFilters, (self.numFilters, 1, -1))
		flattenedFilterLength = self.filterSize*self.filterSize

		for i in range((n - k + 1) * (m - k + 1) - flattenedFilterLength +1):
			convMatrices[:, i, i:i+flattenedFilterLength] = flatenedFilters
		return convMatrices

	#Computes the basis filters that when combined form the convolutional filters (structured as matrix operators)
	def computeConvBasisFilters(self):
		frequencies = np.zeros((self.numBasisFilters, self.filterSize, self.filterSize))
		for i in range(self.filterSize):
			for j in range(self.filterSize):
				frequencies[i*self.filterSize+j, i, j] = 1
		filters2D = fftpack.idct(fftpack.idct(frequencies, axis=1), axis=2)
		#must flatten filters and convert to convolutional matrices
		assert(self.windowSizeX > self.filterSize and self.windowSizeY > self.filterSize)
		n = self.windowSizeY
		m = self.windowSizeX
		k = self.filterSize
		convMatrices = np.zeros((self.numFilters, (n - k + 1) * (m - k + 1), n*m))
		flatenedFilters = np.zeros((self.numFilters, self.filterSize, self.filterSize))
		flatenedFilters[:, :self.filterSize, :self.filterSize] = self.filters
		flatenedFilters = np.reshape(flatenedFilters, (self.numFilters, 1, -1))
		flattenedFilterLength = self.filterSize*self.filterSize

		for i in range((n - k + 1) * (m - k + 1) - flattenedFilterLength +1):
			convMatrices[:, i, i:i+flattenedFilterLength] = flatenedFilters
		return convMatrices

	#The derivative of the energy term, see paper for details 
	def deltaE(self, x, noisyFlatImage):
		filters = self.computeConvFilters()
		sum = 0
		conv1 = np.dot(filters, x)
		phi_prime = self.phiPrime(conv1)
		conv2 = self.alpha*np.dot(filters.transpose((0, 2, 1)), phi_prime)
		deltaE = np.sum(conv2, axis=0) + x - noisyFlatImage
		return deltaE

	#The energy term, see paper for details 
	def E(self, image):
		filters = self.computeConvFilters()
		conv = filters.dot(image)
		reutrn -np.sum(np.log(self.phi(conv)))

	#A function used in the FoE model, see paper for details
	def phi(self, x):
		return self.alpha*np.log(1+x**2)

	#The first derivative of the phi function 
	def phiPrime(self, input):
		return self.alpha*2*input/(1+input**2)

	#The second derivative of the phi function
	def phiPrimePrime(self, input):
		#TODO: second derivative of phi prime
		print('phiPrimePrime not implemented')
		return input

	#The diagonal matrix required for optimization, see paper for details
	def diagonal(self, x):
		filters = self.computeConvFilters()
		phiPP = phiPrimePrime(np.dot(filters, x))
		d = np.zeros((self.numFilters, phiPP.size[1]))
		for n in range(self.numFilters):
			d[n, :, :] = np.diagflat(phiPP[n, :])
		return d

	#The hessian matrix required for optimization, see paper for details
	def hessian(self, x):
		filters = self.computeConvFilters()
		d = diagonal(x)
		conv1 = np.dot(filters, d)
		conv2 = np.dot(filters.transpose((0, 2, 1)), conv1)
		return sum(self.alpha*conv2, axis=0) + np.identity(conv2.shape[1])

	#The main function for training the FoE model, one sample at a time 
	def train(self, noisyImageBatch, trueImageBatch):
		#compute gradients for batch, only updating at the end
		deltaAlpha = np.zeros((self.numFilters))
		deltaBeta = np.zeros((self.numBasisFilters))
		for index in range(noisyImageBatch.shape()[0]):
			#flatten images 
			noisyFlat = np.reshape(noisyImageBatch[index, :, :], (-1, 1))
			trueFlat = np.reshape(trueImage[index, :, :], (-1, 1))
			#estimate image 
			guess = np.zeros((self.windowSizeY*self.windowSizeX))
			estimate = optimize.newton(lambda x : self.deltaE(x, noisyFlat), guess)

			#compute loss gradient (see Xu equations 7-8)
			#compute diagonal D
			d = diagonal(guess)
			#compute Hessian 
			h = hessian(guess)
			hInv = np.linalg.tensorinv(h)
			#compute basis filters 
			b = self.basisFilters
			bt = b.transpose((0, 2, 1))
			#compute learned filters
			f = self.filters
			ft = f.transpose((0, 2, 1))

			#update alpha and beta weights according to noisy and true images
			deltaAlpha += -(ft.dot(phiPrime(np.dot(f, guess)))).transpose((0, 2, 1)).dot(hInv).dot(guess-trueImage)

			deltaBeta += -(bt.dot(phiPrime(np.dot(f, guess))) + ft.dot(d).dot(b).dot(guess)).transpose((0, 2, 1)).dot(hInv).dot(guess-trueImage)

		#update weights and filters
		deltaBeta = deltaBeta.reshape((self.numFilters, self.numFilters))
		self.alpha = self.alpha - self.alphaStepSize*deltaAlpha
		self.beta = self.beta - self.betaStepSize*deltaBeta
		self.filters = self.computeConvFilters()



