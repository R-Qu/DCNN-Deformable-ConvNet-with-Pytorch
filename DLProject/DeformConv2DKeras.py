from __future__ import division

from keras.backend import tf
from keras import initializers
from keras.layers import Conv2D
from tf_interpolate_2 import batch_map_offsets

class Conv2DOffset(Conv2D):
    """ConvOffset2D"""

    def __init__(self, filters, init_normal_stddev=0.05, **kwargs):
        """Init"""

        self.filters = filters
        super(Conv2DOffset, self).__init__(
            18, (3, 3), padding='same', use_bias=False,
            #kernel_initializer=initializers.Zeros(),
            #kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=init_normal_stddev, seed=123),
            **kwargs
        )

    def call(self, x):
        x_shape = x.get_shape()
        offsets = super(Conv2DOffset, self).call(x)
        #offsets *= 10

        channels = int(offsets.get_shape()[3].value)
        n_batches = tf.shape(offsets)[0]

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        ind_shuffle = tf.concat([tf.range(0, channels, 2),
                                tf.range(1, channels + 1, 2)], axis=0)

        #ind_shuffle = tf.expand_dims(ind_shuffle, axis=0)
        #ind_shuffle = tf.expand_dims(ind_shuffle, axis=0)
        #ind_shuffle = tf.tile(ind_shuffle, [input_w, input_h, 1])

        offsets = tf.gather(offsets, ind_shuffle, axis=3)
        # ------------------------------------------------------------------------
        #x = tf.transpose(x, [0, 3, 1, 2])
        #x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))
        #offsets = tf.resampler(x, offsets)
        offsets = batch_map_offsets(x, offsets)
        #offsets = tf.reshape(x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2])))
        #offsets = tf.transpose(x, [0, 2, 3, 1])
        offset_shape = offsets.get_shape()
        num_channels = offset_shape[1].value
        height = offset_shape[2].value
        width = offset_shape[3].value
        f_offset = [tf.reshape(offsets[..., ind:ind + 3], 
                               (-1, num_channels, height, width * 3))
                   for ind in range(0, 9, 3)]
        f_offset = tf.concat(f_offset, axis=-1)
        f_offset = tf.reshape(f_offset, (-1, num_channels, height * 3, width * 3))
        f_offset = tf.transpose(f_offset, (0, 2, 3, 1))
        return f_offset

    def index_add():
        x_slice

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 3, input_shape[2] * 3, input_shape[3])


