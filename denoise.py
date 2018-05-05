#The main function for image denoiser
#This method evaluates a previously trained model on MRI images, 
#collects statistics on performance, and saves the resutls.
import util

def main():
	#load trained FoE model from file
	##foe = FoE.load(foeModelFilename)
	#load test images 
	imagesDirectory = '/Users/rvansoelen/Documents/mriDenoiser/data'
	images = util.loadImages(imagesDirectory)

	#for each test image
	
	for image in images:
		#for each window in image
		windows = util.segmentImage(image)
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
	


if __name__== '__main__':
	main()