# 3. Import libraries and modules
import numpy as np
#import tensorflow as tf
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, GlobalAvgPool2D
from keras.utils import np_utils
from keras.datasets import cifar10
from Conv2DOffset import Conv2DOffset
from deeplab import Deeplabv3

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
    #############################

    # 7. Define model architecture
    model = Deeplabv3(input_shape=(32, 32, 3), classes=10, trainer=True)
    
    if (preLoad):
        model.load_weights('cifar_weights.h5', by_name=True)

    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
 
    # 9. Fit model on training data
    model.fit(X_train, Y_train, 
              batch_size=32, epochs=10, verbose=1)
    
    if (saveWeights):
        model.save_weights('cifar_weights.h5')
    # 10. Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)

if __name__ == "__main__":
    main()

