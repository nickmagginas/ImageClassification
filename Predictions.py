from keras.models import load_model
from ImagePreprocessor import ImagePreprocessor
import numpy as np
from joblib import load, dump
import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model


IMAGES = [f'testing_dataset/{i}.png' for i in range(1,101)]
BASE_MODEL = ResNet50(weights = 'imagenet')
MODEL = Model(inputs = BASE_MODEL.input, outputs = BASE_MODEL.get_layer('avg_pool').output)
architechture = 'SVM'

''' get predictions for SVMS and CNNS and write them to files '''
''' Preprocessing exactly the same as for training '''

def getSample(path):
	data = image.load_img(path, target_size = (224,224))
	data = image.img_to_array(data)
	return data

def getImages():
	return np.array([getSample(image) for image in IMAGES])

def getFeaturesSVM(): return MODEL.predict(getImages(), verbose = 1)

def getFeatures(gray = False, norm = False):
	    images = np.array([ImagePreprocessor.readImage(image, grayscale = gray, normalize = norm) for image in IMAGES])
	    return np.array([ImagePreprocessor.resizeImage(image) for image in images])

if architechture == 'CNN':
	features = getFeatures()
	features = features.reshape((len(features), 128, 128, 3))
else:
	features = getFeaturesSVM()
	features = features.reshape((len(features), 2048))

for task in range(1,6):
	if architechture == 'CNN':
		model = load_model(f'models/Task{task}.h5')
		predictions = model.predict(features)
		if task == 1: predictions = [np.argmax(prediction) for prediction in predictions]
		else: predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]
		data = pd.DataFrame({'fileName': IMAGES, 'Prediction': predictions})
		data.to_csv(f'files/Task{task}.csv', index = False)
	elif architechture == 'SVM' and task != 1:
		model = load(f'models2/{task-2}.joblib')
		predictions = model.predict(features)
		data = pd.DataFrame({'fileName': IMAGES, 'Prediction': predictions})
		data.to_csv(f'files/Task{task}SVM.csv', index = False)


