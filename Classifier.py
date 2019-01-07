import numpy as np
from ImagePreprocessor import ImagePreprocessor
from sklearn.model_selection import train_test_split as split
from keras.models import Sequential
from keras import optimizers
from keras.models import load_model as load
from keras.utils import to_categorical as onehot
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt


def createModel(multiclass = False):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding = 'same', input_shape = (128,128,3), activation = 'relu'))
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3,3), activation = 'relu')) 
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3,3), activation = 'relu')) 
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(1024,activation = 'relu'))
    model.add(Dropout(0.50))
    if multiclass:  
        model.add(Dense(6, activation = 'softmax'))
    else:
        model.add(Dense(1, activation = 'sigmoid'))
    return model

def train():
    columns = [f'Unnamed: {i}' for i in range(3,4)]
    for index,column in enumerate(columns):
        features, labels = ImagePreprocessor.createDataset(column, grayscale = False, normalize = True)
        features = np.reshape(features, (len(features), 128, 128, 3))
        if index == 1: labels = onehot(labels)
        else: labels = [0 if label == '-1' else int(label) for label in labels]
        features, testX, labels, testY = split(features, labels, test_size = 0.2)
        model = createModel(multiclass = (column == 'Unnamed: 1'))
        callbacks = [EarlyStopping(monitor = 'val_loss', restore_best_weights= True, patience = 2), ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 1)]
        #callbacks.append(ModelCheckpoint(f'models/Task{index+1}.h5'))
        loss = 'categorical_crossentropy' if index == 1 else 'binary_crossentropy'
        model.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])
        history = model.fit(features, labels, batch_size = 32, epochs = 20, validation_data = (testX, testY), callbacks = callbacks, class_weight = 'auto')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('CNN Loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        #model.save(f'models2/Task{index+1}.h5')

if __name__ == '__main__':
    train()
