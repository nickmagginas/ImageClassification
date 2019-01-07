# ImageClassification
Machine Learning Assignment on Facial Recognition

# Library Dependancies
Sklearn, SkImage, Keras, Tensorflow, DLib, pandas, numpy, joblib, Open-CV(cv2).

# Additional Dependencies
Some functions such as saving models and loading them back might require some additional libraries such as h5py for keras.

# Code
There are 4 distinct files:   
-- Classifier.py: Trains CNN for all tasks  
-- tranferLearning.py: Feature extraction with ResNet and train linear SVM   
-- ImagePreprocessor.py: Class for ImagePreproccessing prior to training   
-- Predictions.py: For writing final predictions to CSV   

# GuideLines
Preferably you have write and read permissions since many files write output to new directories which might not exist in your system. If not, you can manually create the directories:      
-- models For CNN   
-- models2  For SVM   
-- files For any Files   

Most Importantly the dataset must be present in a folder called dataset and contain all samples. Further for generation of predicted labels there must also be a testing_dataset directory.     
--- dataset !!!   
--- testing_dataset !!!   

# Submitted Zip
The submitted zip contains all relevant files including all the models, dataset and helpful files. You might wish to copy the dropIndices numpy file which specifies the indices to be droped since it takes quite a lot of time to create the dataset for the first time if this file does not exist beacasue we have to apply the HOG classifier to each individual image.
