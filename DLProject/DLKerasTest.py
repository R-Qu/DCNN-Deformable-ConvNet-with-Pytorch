# 3. Import libraries and modules
import numpy as np
#import tensorflow as tf
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, GlobalAvgPool2D
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.backend import tf
from keras.preprocessing.image import ImageDataGenerator
from Conv2DOffset import *
from Pool2DOffset import *

def main(): 
    # 4. Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
 
    # 5. Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    image_scaler = ImageDataGenerator(
        zoom_range=(1, 2.5),
        width_shift_range=0.2,
        height_shift_range=0.2
        )
    image_zoomer = ImageDataGenerator(
        zoom_range=(2.5, 2.5),
        width_shift_range=0,
        height_shift_range=0
        )
    
    X_train /= 255
    X_test /= 255
    
    # 6. Preprocess class labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    batch_s = 32
    n_train = 50000
    n_test = 10000

    steps_pe = int(np.ceil(n_train / batch_s))
    val_steps = int(np.ceil(n_test / batch_s))


    X_test_scaled = image_scaler.flow(X_test, Y_test, batch_size=batch_s, shuffle=True)
    X_train_zoomed = image_zoomer.flow(X_train, Y_train, batch_size=batch_s, shuffle=True)
    X_train_scaled = image_scaler.flow(X_train, Y_train, batch_size=batch_s, shuffle=True)
    ########################## TRAIN ITERATION 1 ##########################

    ##### PROGRAM CONSTANTS #####
    preLoad = False
    saveWeights = True
    trainModel = True
    #############################

    # 7. Define model architecture
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3), trainable = trainModel, name='conv1'))
    #model.add(Conv2DOffset(32, name='convoffset1'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2), name='conv2', trainable = trainModel))
    #model.add(Conv2DOffset(64, name='convoffset2'))
    #model.add(Pool2DOffset(64, name='pooloffset1'))
    model.add(MaxPooling2D(pool_size=(2,2), trainable = trainModel, name='pool'))
    
    model.add(Flatten(trainable = trainModel, name='flat'))
    model.add(Dense(64, activation='relu', trainable = trainModel, name='dense1'))
    model.add(Dense(10, activation='softmax', trainable = trainModel, name='dense2'))

    if (preLoad):
        model.load_weights('cifar_weights.h5', by_name=True)
    
    # 8. Compile model
    #model.compile(loss='categorical_crossentropy',
    #              optimizer='sgd',
    #              metrics=['accuracy'])
    #print("\nTraining our initial model for 20 epochs...\n\n")
    ## 9. Fit model on training data
    #model.fit(X_train, Y_train, 
    #          batch_size=batch_s, epochs=20, verbose=1)
    
    #if (saveWeights):
    #    model.save_weights('cifar_weights.h5')
    # 10. Evaluate model on test data
    #score = model.evaluate(X_test, Y_test, verbose=0)

    ########################## TRAIN ITERATION 2 ##########################

    ##### PROGRAM CONSTANTS #####
    preLoad = True
    saveWeights = False
    trainModel = True
    #############################

    # 7. Define model architecture
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3), trainable = trainModel, name='conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2), name='conv2'))
    #model.add(Pool2DOffset(64, name='pooloffset1'))
    model.add(MaxPooling2D(pool_size=(2,2), trainable = trainModel, name='pool'))
    
    model.add(Flatten(trainable = trainModel, name='flat'))
    model.add(Dense(64, activation='relu', trainable = trainModel, name='dense1'))
    model.add(Dense(10, activation='softmax', trainable = trainModel, name='dense2'))

    if (preLoad):
        model.load_weights('cifar_weights.h5', by_name=True)

    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    print("\nTraining our model with regular convolution for 10 epochs...\n\n")
    # 9. Fit model on training data
    model.fit_generator(X_train_zoomed, steps_per_epoch=steps_pe,
         epochs=1, verbose=1, validation_steps=val_steps)
    
    if (saveWeights):
        model.save_weights('cifar_weights.h5')
    # 10. Evaluate model on test data
    score1 = model.evaluate(X_test, Y_test, verbose=0)
    print("Score of our model with regular convolution is(normal test data): Loss: "
         + str(score1[0]) + "\tAccuracy: " + str(score1[1]))
    score2 = model.evaluate_generator(X_test_scaled, verbose=0, steps=val_steps)
    print("Score of our model with regular convolution is(scaled test data): Loss: "
          + str(score2[0]) + "\tAccuracy: " + str(score2[1]))

    ########################## TRAIN ITERATION 3 ##########################

    ##### PROGRAM CONSTANTS #####
    preLoad = True
    saveWeights = False
    trainModel = True
    #############################

    # 7. Define model architecture
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3), trainable = trainModel, name='conv1'))
    model.add(Conv2DOffset(32, name='convoffset1'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2), trainable = trainModel, name='conv2'))
    model.add(Pool2DOffset(64, name='pooloffset1'))
    model.add(MaxPooling2D(pool_size=(2,2), trainable = trainModel, name='pool'))
 
    model.add(Flatten(trainable = trainModel, name='flat'))
    model.add(Dense(64, activation='relu', trainable = trainModel, name='dense1'))
    model.add(Dense(10, activation='softmax', trainable = trainModel, name='dense2'))

    if (preLoad):
        model.load_weights('cifar_weights.h5', by_name=True)

    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    print("\nTraining our model with deformable convolution for 10 epochs...\n\n")
    # 9. Fit model on training data
    model.fit_generator(X_train_zoomed, steps_per_epoch=steps_pe, 
              epochs=1, verbose=1, validation_steps=val_steps)
    
    if (saveWeights):
        model.save_weights('cifar_weights.h5')
    # 10. Evaluate model on test data
    score1 = model.evaluate(X_test, Y_test, verbose=0)
    print("Score of our model with deformable convolution is(normal test data): Loss: " 
          + str(score1[0]) + "\tAccuracy: " + str(score1[1]))
    score2 = model.evaluate_generator(X_test_scaled, verbose=0, steps=val_steps)
    print("Score of our model with deformable convolution is(scaled test data): Loss: " 
          + str(score2[0]) + "\tAccuracy: " + str(score2[1]))

if __name__ == "__main__":
    main()
