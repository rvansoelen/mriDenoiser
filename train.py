#trains the foe model
#This function trains the Field of Experts Model. It loads the training images, 
#updates the FoE model, and save the model to a file (using the pickling technique)
import foe, gmm, mapEstimator
from scipy import ndimage

def main():
	filenames = ['outoutPlaceholder']
	#load noisy and ground truth images
	images = []
	for filename in filenames:
		images.append(ndimage.imread(filename))

	#create initial foe model
	foe = FoE()

	#for each epoch (if more than one)
	numEpochs = 10
	for epoch in range():
		#for each image:
		for image in images:
			#call training function of foe model
			foe.train(image)

	#save model to file
	foe.save('output.foe')
