#The main function for image denoiser
#This method evaluates a previously trained model on MRI images, 
#collects statistics on performance, and saves the resutls.

def main() {
	#load trained FoE model from file
	foe = loadFoEModel(foeModelFilename)
	#load test images 
	images = loadImages(imagesDirectory)
	#for each test image
	for image in images:
		#for each window in image
		windows = segmentImage(image)
		for window in windows:
			#calculate prior probability from FoE
			prior = foe.getPrior()

			#initialize weights of GMM
			gmm = GMM(window)

			#compose MAP probability to maximize
			map = MAPEstimator(gmm, prior)

			#call maximization algorithm to produce denoised image
			denoisedWindow = map.estimate()

		#stitch together windows

		#calculate accuracy of denoise

	#save results
}

def loadFoEModel(foeModelFilename):
	#load model from file

def loadImages(imagesDirectory):
	#load images from directory 

def segmentImage(image):
	#divide images into small windows 

	#return windows