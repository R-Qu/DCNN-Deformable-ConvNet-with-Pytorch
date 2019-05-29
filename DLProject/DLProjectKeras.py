# 3. Import libraries and modules
import numpy as np
#import tensorflow as tf
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, GlobalAvgPool2D
from keras.utils import np_utils
from keras.datasets import cifar10
from DeformConv2DKeras import *

def main(): 
    # 4. Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
 
    # 5. Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # 6. Preprocess class labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    
    ##### PROGRAM CONSTANTS #####
    preLoad = False
    saveWeights = False
    trainModel = True
    deformable = True
    #############################

    # 7. Define model architecture
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3), trainable = trainModel, name='conv1'))
    #model.add(Conv2D(64, (3, 3), activation='relu', name='conv2'))
    if (deformable):
        model.add(Conv2DOffset(32, name='convoffset'))
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='valid'))
    else:
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), trainable = trainModel, name='pool'))
 
    model.add(Flatten(trainable = trainModel, name='flat'))
    model.add(Dense(32, activation='relu', trainable = trainModel, name='dense1'))
    model.add(Dense(10, activation='softmax', trainable = trainModel, name='dense2'))

    if (preLoad):
        model.load_weights('cifar_weights.h5', by_name=True)

    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
 
    # 9. Fit model on training data
    model.fit(X_train, Y_train, 
              batch_size=32, epochs=10, verbose=1, validation_data=(X_test, Y_test))
    
    if (saveWeights):
        model.save_weights('cifar_weights.h5')
    # 10. Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)

if __name__ == "__main__":
    main()

