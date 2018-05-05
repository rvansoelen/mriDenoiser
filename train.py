#trains the foe model
#This function trains the Field of Experts Model. It loads the training images, 
#updates the FoE model, and save the model to a file (using the pickling technique)
import foe
import util

def main():
	noisyDirectory = '/Users/rvansoelen/Documents/mriDenoiser/data/noisyData'
	groundTruthDirectory = '/Users/rvansoelen/Documents/mriDenoiser/data/groundTruth'

	#load noisy and ground truth images
	noisyBatch = util.loadImagesAsSegments(noisyDirectory)
	truthBatch = util.loadImagesAsSegments(groundTruthDirectory)
	#assume they are ordered correctly


	#create initial foe model
	FoE = foe.FoE()

	#for each epoch (if more than one)
	numEpochs = 1
	for epoch in range(numEpochs):
		#call training function of foe model
		FoE.train(noisyBatch, truthBatch)


	#save model to file
	FoE.save('output.foe')


if __name__ == '__main__':
	main()