from __future__ import division

from keras.backend import tf
from keras import initializers
from keras.layers import Dense
from tf_interpolate import batch_map_offsets

class Pool2DOffset(Dense):
    """ConvOffset2D"""

    def __init__(self, units, **kwargs):
        """Init"""

        self.units = units
        super(Pool2DOffset, self).__init__(
            self.units * 2, use_bias=False,
            #kernel_initializer=initializers.Zeros(),
            #kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=init_normal_stddev, seed=123),
            **kwargs
        )

    def call(self, x):
        x_shape = x.get_shape()
        offsets = super(Pool2DOffset, self).call(x)
        offsets = tf.transpose(offsets, [0, 3, 1, 2])
        offsets = tf.reshape(offsets, (-1, int(x_shape[1]), int(x_shape[2]), 2))
        n_batches = tf.shape(offsets)[0]
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 1))

        

        #offsets = tf.resampler(x, offsets)
        x = batch_map_offsets(x, offsets)
        x = tf.reshape(x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2])))
        x = tf.transpose(x, [0, 2, 3, 1])
        return x

    def index_add():
        x_slice

    def compute_output_shape(self, input_shape):
        return input_shape
