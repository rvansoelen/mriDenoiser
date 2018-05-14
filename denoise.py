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
	foeModelFilename = './output/model_3pn_final.foe'
	foeModel = foe.load(foeModelFilename)
	#load test images 
	noisyDirectory = './data/noisyData/3pn'
	groundTruthDirectory = './data/groundTruthData'
	noisyImages = util.loadImagesAsSegments(noisyDirectory)
	trueImages = util.loadImagesAsSegments(groundTruthDirectory)
	saveImageResults = False

	#for each test image
	map = mapest.MAPEstimator(foeModel)
	errorList = []
	for noisyImage, trueImage in itertools.izip(noisyImages, trueImages):
		#for each window in image
		noisySegs = util.segmentImage(noisyImage)
		trueSegs = util.segmentImage(trueImage)
		for noisySeg, trueSeg in itertools.izip(noisySegs, trueSegs):
			#initialize weights of GMM
			gmmModel = gmm.GMM(noisySeg)
			map.gmm = gmmModel

			#call maximization algorithm to produce denoised image
			denoisedWindow = map.estimate()

			#calculate accuracy of denoise
			err = np.mean((noisySeg-trueSeg)**2)
			print err
			errorList.append(err)
			plt.subplot(3, 1, 1)
			plt.imshow(noisySeg.reshape((util.windowHeight, util.windowWidth)), cmap='gray')
			plt.subplot(3, 1, 2)
			plt.imshow(denoisedWindow.reshape((util.windowHeight, util.windowWidth)), cmap='gray')
			plt.subplot(3, 1, 3)
			plt.imshow(trueSeg.reshape((util.windowHeight, util.windowWidth)), cmap='gray')
			plt.show()
				
		
		if saveImageResults:
			#stitch together windows
			#TODO
			print 'Image stitching needs to be implemented'

	#save results
	avgError = np.mean(errorList)
	print 'Average Mean Squared Error: ', avgError
	with open('errorList.pckl','wb') as fid:
		pickle.dump(self,fid)



if __name__== '__main__':
	main()