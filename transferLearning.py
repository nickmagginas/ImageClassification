from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.models import Model
import numpy as np
from ImagePreprocessor import ImagePreprocessor
from sklearn.svm import SVC
from joblib import dump, load

''' ResNet Base Model only up to second to last layer '''
BASE_MODEL = ResNet50(weights = 'imagenet')
MODEL = Model(inputs = BASE_MODEL.input, outputs = BASE_MODEL.get_layer('avg_pool').output)
IMAGES = [f'dataset/{i}.png' for i in range(1,5001)]

def getSample(path):
	''' keras read image '''
	data = image.load_img(path, target_size = (224,224))
	data = image.img_to_array(data)
	return data

def getImages():
	''' remove some images that are not compatible and read data for the rest '''
	remove = np.load('files/dropIndices.npy') 
	remove[49] = False
	remove[3164] = False
	np.save('files/dropIndices.npy', remove)
	remove = np.load('files/dropIndices.npy')
	return np.array([getSample(image) for index,image in enumerate(IMAGES) if remove[index]])

def getFeatures(): return MODEL.predict(getImages(), verbose = 1)
''' return all of the features '''

features = getFeatures()
features = features.reshape((len(features), 2048))
''' reshape them to 2D array '''
for index,column in enumerate([f'Unnamed: {i}' for i in range(2,6)]):
	''' get Labels for each task '''
	labels = ImagePreprocessor.getLabels(column)
	labels = [[0] if label == '-1' else [1] for label in labels]
	''' split to test and train '''
	X, testX, Y, testY = train_test_split(features, labels, test_size = 0.2)
	''' SVM learning '''
	clf = SVC(kernel = 'linear')
	clf.fit(X, Y)
	''' Save '''
	dump(clf, f'models2/{index}.joblib')
	''' Load '''
	clf = load(f'models2/{index}.joblib')
	''' Test Accuracy '''
	print(clf.score(testX, testY))
