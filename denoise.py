#The main function for image denoiser
#This method evaluates a previously trained model on MRI images, 
#collects statistics on performance, and saves the resutls.
import util
import pdb
import numpy as np
import pickle
import foe, gmm
import mapEstimator as mapest
import itertools
import matplotlib.pyplot as plt

def main():
	#load trained FoE model from file
	foeModelFilename = './output/3pn_pt4/model_3pn_final.foe'
	foeModel = foe.load(foeModelFilename)
	#load test images 
	noisyDirectory = './data/noisyData/3pn'
	groundTruthDirectory = './data/groundTruthData'
	noisyImages = util.loadImagesAsSegments(noisyDirectory)
	trueImages = util.loadImagesAsSegments(groundTruthDirectory)
	showImageResults = True

	#for each test image
	map = mapest.MAPEstimator(foeModel)
	errorHistory = []
	numSamples = 20
	for noisyImage, trueImage in itertools.izip(noisyImages, trueImages):
		#for each segment in image
		noisySegs = util.segmentImage(noisyImage)
		trueSegs = util.segmentImage(trueImage)
		for noisySeg, trueSeg in itertools.izip(noisySegs[:numSamples], trueSegs[:numSamples]):
			#initialize weights of GMM
			gmmModel = gmm.GMM(noisySeg)
			map.gmm = gmmModel

			#call maximization algorithm to produce denoised image
			denoisedSegment = map.estimate()

			#calculate accuracy of denoise
			err = np.mean((noisySeg-trueSeg)**2)
			errorHistory.append(err)

			#show the denoised segment (if enabled)
			if showImageResults:
				plt.subplot(1, 3, 1)
				plt.imshow(noisySeg.reshape((util.segmentHeight, util.segmentWidth)), cmap='gray')
				plt.title('Noisy Segment')
				plt.subplot(1, 3, 2)
				plt.imshow(denoisedSegment.reshape((util.segmentHeight, util.segmentWidth)), cmap='gray')
				plt.title('Reconstructed Segment')
				plt.subplot(1, 3, 3)
				plt.imshow(trueSeg.reshape((util.segmentHeight, util.segmentWidth)), cmap='gray')
				plt.title('True Noise-Free Segment')
				plt.show()

	#save results
	avgError = np.mean(errorHistory)
	print 'Average Mean Squared Error: ', avgError
	with open('errorHistory.pckl','wb') as fid:
		pickle.dump(errorHistory,fid)



if __name__== '__main__':
	main()