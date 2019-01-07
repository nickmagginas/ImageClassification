import os
if not os.path.isfile('files/dropIndices.npy'):
	import dlib
	CLASSIFIER = dlib.get_frontal_face_detector()
import pandas as pd
import numpy as np
import cv2
from skimage.transform import resize

IMAGES = [f'dataset/{i}.png' for i in range(1,5001)]

''' Collection of helpfull functions '''

class ImagePreprocessor:

	@staticmethod
	def readImage(path, grayscale = False, normalize = False):
		''' read Imnage using Open-CV '''
		image = cv2.imread(path)
		if grayscale: image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		if normalize: image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		return image

	@staticmethod
	def convolve(subimage, kernel):
		''' Convolution of kernel and subimage '''
	    if subimage.shape != kernel.shape:
	        raise TypeError('Dimension Mismatch')
	    subimage, kernel = list(subimage), list(kernel)
	    return np.sum([subimage[index]*value for index,value in enumerate(kernel)])

	@staticmethod
	def applyKernel(image, kernel):
		''' kernel application to full image '''
	    horizontalPad, verticalPad = tuple(map(lambda x: 0 if x == 1 else x%2, kernel.shape))
	    width, height = tuple(map(lambda x: x - 1,image.shape))
	    image = np.pad(image, [(horizontalPad,horizontalPad), (verticalPad,verticalPad)], mode = 'constant', constant_values = 0)
	    convolved = np.zeros((width+1,height+1))
	    for column,row in np.ndindex((width,height)):
	        if verticalPad: subimage = image[column+verticalPad,row+horizontalPad:row+kernel.shape[1]]
	        else: subimage = image[column+verticalPad:column+kernel.shape[0],row+horizontalPad]
	        subimage = subimage.reshape(kernel.shape)
	        convolved[column][row] = ImagePreprocessor.convolve(subimage,kernel)
	    return convolved

	@staticmethod
	def cropImage(image, v1 = (32,32), v2 = (224,224)):
		''' Croping '''
	    return image[v1[0]:v2[0], v1[1]:v2[1]]

	@staticmethod
	def resizeImage(image, ratio = (2,2)):
		''' Lower Resolution '''
		return resize(image, (image.shape[0]/ratio[0], image.shape[1]/ratio[1]))

	@staticmethod
	def maxPool(image, depth = 3):
		''' 2 by 2 filters for 192 * 192 grayscale Image '''
	    downsampled = np.zeros((96,96,depth))
	    for column, row in np.ndindex((downsampled.shape[0], downsampled.shape[1])):
	        window = image[column*2:column*2+2, row*2:row*2+2].reshape((4,depth))
	        maximum = 0
	        for pixel in window: 
	            if sum(pixel) >= maximum:
	                maximum = sum(pixel)
	                downsampled[column][row] = pixel
	    downsampled = cv2.normalize(downsampled, None, 0, 1, cv2.NORM_MINMAX)
	    return downsampled

	@staticmethod
	def getMagnitude(image, kernel):
		''' Magnitude of gradients '''
	    horizontal = ImagePreprocessor.applyKernel(image,kernel)
	    vertical = ImagePreprocessor.applyKernel(image, kernel.T)
	    return np.sqrt(horizontal**2 + vertical**2)

	@staticmethod
	def filterImage(path): return True if CLASSIFIER(cv2.imread(path), 1) else False
	''' See whether image contains a face '''

	@staticmethod
	def filterDataset(): return list(map(ImagePreprocessor.filterImage, IMAGES)) 
	''' apply to the whole dataset '''

	@staticmethod
	def recordIndinces(): np.save('files/dropIndices', np.array(ImagePreprocessor.filterDataset()))
	''' write indices that should be excluded '''

	@staticmethod
	def getFeatures(gray = False, norm = False):
		''' read rest of the images '''
	    discard = np.load('files/dropIndices.npy')
	    images = np.array([ImagePreprocessor.readImage(image, grayscale = gray, normalize = norm) for index,image in enumerate(IMAGES) if discard[index]])
	    return np.array([ImagePreprocessor.resizeImage(image) for image in images])

	@staticmethod
	def getLabels(feature):
		''' get labels from csv for specific column '''
	    attributes = pd.read_csv('files/attribute_list.csv')[feature].values[1:]
	    discard = np.load('files/dropIndices.npy')
	    return [attribute for index,attribute in enumerate(attributes) if discard[index]] 

	@staticmethod
	def filterMislabels(features,labels):
		''' filter -1 in hair color which should not be there '''
	    features = [feature for index,feature in enumerate(features) if labels[index] != '-1']
	    labels = list(filter(lambda x: x != '-1', labels))
	    return features,labels

	@staticmethod
	def createDataset(attribute, grayscale = False, normalize = False):
		''' return the whole features and labels for a single task '''
		if not os.path.isfile('files/dropIndices.npy'): ImagePreprocessor.recordIndinces()
		features = ImagePreprocessor.getFeatures(gray = grayscale, norm = normalize)
		labels = ImagePreprocessor.getLabels(attribute)
		if attribute == 'Unnamed: 1': return ImagePreprocessor.filterMislabels(features,labels)
		return features, labels