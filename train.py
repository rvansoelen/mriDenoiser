#trains the foe model
#This function trains the Field of Experts Model. It loads the training images, 
#updates the FoE model, and save the model to a file (using the pickling technique)
import foe
import util
import random
import pdb
import time
import matplotlib.pyplot as plt

def main():
	#noisyDirectory = '/Users/rvansoelen/Documents/mriDenoiser/data/noisyData'
	#groundTruthDirectory = '/Users/rvansoelen/Documents/mriDenoiser/data/groundTruthData'
	noiseLevel = '3pn'
	noisyDirectory = './data/noisyData/'+noiseLevel
	groundTruthDirectory = './data/groundTruthData'
	outputDirectory = './output/3pn_pt4/'

	#load noisy and ground truth images
	noisySegs = util.loadImagesAsSegments(noisyDirectory)
	truthSegs = util.loadImagesAsSegments(groundTruthDirectory)
	#assume they are ordered correctly
	segPairs = zip(noisySegs, truthSegs)

	#create initial foe model
	FoE = foe.FoE()

	#for each epoch (if more than one)
	numEpochs = 1
	batchSize = 10
	numBatches = len(segPairs)/batchSize
	print 'Starting...'
	start = time.time()
	for epoch in range(numEpochs):
		print 'Epoch: ', epoch
		random.shuffle(segPairs)
		for batchNumber, segPairBatch in enumerate(util.batch(segPairs, batchSize=batchSize)):
			if batchNumber %100==0: 
				lap = time.time() - start
				print 'Batch ', batchNumber, ' out of ', numBatches, ': ', lap, ' s'
				start = time.time()
				FoE.save(outputDirectory+'model_'+noiseLevel+'_'+str(batchNumber)+'.foe')
			#call training function of foe model
			#updates the weights only once
			FoE.train(segPairBatch)


	#save model to file
	FoE.save(outputDirectory+'model_'+noiseLevel+'_final.foe')
	
	#Plot error
	plt.plot(FoE.errorHistory)
	plt.xlabel('Training Batch Iteration')
	plt.ylabel('L2 Loss')
	plt.show()


if __name__ == '__main__':
	main()