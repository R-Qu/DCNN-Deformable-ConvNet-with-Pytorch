# 3. Import libraries and modules
import numpy as np
#import tensorflow as tf
np.random.seed(123)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, GlobalAvgPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.datasets import cifar10
from Conv2DOffset import Conv2DOffset
from vgg import vgg16
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

    image_normal = ImageDataGenerator(
        zoom_range=(1, 1),
        )

    image_zoomed = ImageDataGenerator(
        zoom_range=(2, 2),
        )

    ##### PROGRAM CONSTANTS #####
    preLoad = False
    saveWeights = False
    trainModel = True
    batch_s = 32
    n_train = 50000
    n_test = 10000
    #############################
    X_test_zoomed = image_zoomed.flow(X_test, Y_test, batch_size=batch_s, shuffle=True)
    X_train_normal = image_normal.flow(X_train, Y_train, batch_size=batch_s, shuffle=True)
    steps_pe = int(np.ceil(n_train / batch_s))
    val_steps = int(np.ceil(n_test / batch_s))
    #############################

    # 7. Define model architecture

    my_vgg = vgg16(train=trainModel, num_classes=10, input_shape=(32,32,3), deformable=True, normalizer=True, full_model=True,
                 last_pooling=False)
    model = my_vgg.build_model()
    #model = Deeplabv3(input_shape=(32, 32, 3), classes=10, trainer=trainModel, def_conv=True)
    if (preLoad):
        model.load_weights('cifar_weights_vgg_deformed_2.h5', by_name=True)

    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
 
    # 9. Fit model on training data
    history = model.fit_generator(X_train_normal, 
              steps_per_epoch=steps_pe, validation_steps=val_steps, 
              epochs=2, verbose=1, validation_data=X_test_zoomed)
    
    if (saveWeights):
        model.save_weights('cifar_weights_vgg_2.h5')
    # 10. Evaluate model on test data
    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    #plt.title('Model accuracy')
    #plt.ylabel('Accuracy')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()

    ## Plot training & validation loss values
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('Model loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()

    score1 = model.evaluate(X_test, Y_test, verbose=0)
    score2 = model.evaluate_generator(X_test_zoomed, steps=val_steps, verbose=0)
    print(str(score1[0]) + ' ' + str(score1[1]))
    print(str(score2[0]) + ' ' + str(score2[1]))

if __name__ == "__main__":
    main()

