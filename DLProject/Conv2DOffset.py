from __future__ import division

from keras.backend import tf
from keras import initializers
from keras.layers import Conv2D
from tf_interpolate import batch_map_offsets

class Conv2DOffset(Conv2D):
    """ConvOffset2D"""

    def __init__(self, filters, init_normal_stddev=0.05, **kwargs):
        """Init"""

        self.filters = filters
        super(Conv2DOffset, self).__init__(
            self.filters * 2, (3, 3), padding='same', use_bias=False,
            kernel_initializer=initializers.Zeros(),
            #kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=init_normal_stddev, seed=123),
            **kwargs
        )

    def call(self, x):
        x_shape = x.get_shape()
        offsets = super(Conv2DOffset, self).call(x)
        #offsets *= 10
        offsets = tf.transpose(offsets, [0, 3, 1, 2])
        offsets = tf.reshape(offsets, (-1, int(x_shape[1]), int(x_shape[2]), 2))
        n_batches = tf.shape(offsets)[0]
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 1))

        

        #offsets = tf.resampler(x, offsets)
        offsets = batch_map_offsets(x, offsets)
        offsets = tf.reshape(x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2])))
        offsets = tf.transpose(x, [0, 2, 3, 1])
        return offsets

    def index_add():
        x_slice

    def compute_output_shape(self, input_shape):
        return input_shape


