from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from DeformConv2DKeras import Conv2DOffset

class vgg16:
    def __init__(self, train=True, num_classes=10, input_shape=(None,None,3), deformable=False, normalizer=True, full_model=True,
                 last_pooling=False):
        self.num_classes = num_classes
        self.weight_decay = 0.0005
        self.x_shape = input_shape

        self.train = train

        self.deformable = deformable
        self.full_model = full_model
        self.last_pooling = last_pooling
        self.normalizer = normalizer

    def core(self, input_tensor=None):
        
        input_shape=self.x_shape

        # Determine proper input shape
        ##############################
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1
        ##############################
        trainable = self.train
        weight_decay = self.weight_decay
        full_model = self.full_model
        last_pooling = self.last_pooling
        k_reg = regularizers.l2(weight_decay)
        ##### MAIN CHUNK OF VGG ######
        # Block 1#
        x = Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=self.x_shape, 
                   kernel_regularizer=k_reg)(img_input)
        x = self.batch_norm_drop(x, 0.3)
        x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=k_reg, name='block1_conv2')(x)
        x = self.batch_norm(x)

        if(not last_pooling):
            x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)

        # Block 2#
        x = Conv2D(128, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block2_conv1')(x)
        x = self.batch_norm_drop(x, 0.4)
        x = Conv2D(128, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block2_conv2')(x)
        x = self.batch_norm(x)

        if(not last_pooling):
            x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)

        # Block 3#
        x = Conv2D(256, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block3_conv1')(x)
        x = self.batch_norm_drop(x, 0.4)
        x = Conv2D(256, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block3_conv2')(x)
        x = self.batch_norm_drop(x, 0.4)
        x = Conv2D(256, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block3_conv3')(x)
        x = self.batch_norm(x)

        if(not last_pooling):
            x = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(x)

        # Block 4#
        x = Conv2D(512, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block4_conv1')(x)
        x = self.batch_norm_drop(x, 0.4)
        x = Conv2D(512, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block4_conv2')(x)
        x = self.batch_norm_drop(x, 0.4)
        x = Conv2D(512, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block4_conv3')(x)
        x = self.batch_norm(x)

        if(not last_pooling):
            x = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(x)

        # Block 5#
        x = self.deform(x, 512, 1)
        x = Conv2D(512, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block5_conv1')(x)
        x = self.batch_norm_drop(x, 0.4)
        x = self.deform(x, 512, 2)
        x = Conv2D(512, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block5_conv2')(x)
        x = self.batch_norm_drop(x, 0.4)
        x = self.deform(x, 512, 3)
        x = Conv2D(512, (3, 3), padding='same', activation='relu',kernel_regularizer=k_reg, name='block5_conv3')(x)
        x = self.batch_norm(x)

        x = MaxPooling2D(pool_size=(2, 2), name='block5_pool')(x)
        ##############################

        # FC Layer #
        if(full_model):
            x = Flatten()(x)
            x = Dense(512, kernel_regularizer=k_reg)(x)          
            x = self.batch_norm_drop(x, 0.5)
            x = Dense(self.num_classes)(x)
            x = Activation('softmax')(x)

        return (img_input, x)

    
    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        (img_input, x) = self.core()
        model = Model(img_input, x)
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def batch_norm_drop(self, tensor_input, percentage):
        if (self.normalizer):
            x = BatchNormalization()(tensor_input)
            x = Dropout(percentage)(x)
        else:
            x = tensor_input
        return x

    def batch_norm(self, tensor_input):
        if (self.normalizer):
            x = BatchNormalization()(tensor_input)
        else:
            x = tensor_input
        return x

    def deform(self, tensor_input, channel_size, def_num):
        if (self.deformable):
            x = Conv2DOffset(channel_size, name='deform' + str(def_num))(tensor_input)
            x = Conv2D(channel_size, kernel_size=(3, 3), strides=(3, 3), padding='valid')(x)
        else:
            x = tensor_input
        return x