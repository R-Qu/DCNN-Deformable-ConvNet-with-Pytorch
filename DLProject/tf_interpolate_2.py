import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates
from keras.backend import tf

def tf_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates"""

    assert order == 1

    coords_tl = tf.cast(tf.floor(coords), 'int32')
    coords_br = tf.cast(tf.ceil(coords), 'int32')
    coords_bl = tf.stack([coords_tl[:, 0], coords_br[:, 1]], axis=1)
    coords_tr = tf.stack([coords_br[:, 0], coords_tl[:, 1]], axis=1)

    vals_tl = tf.gather_nd(input, coords_tl)
    vals_br = tf.gather_nd(input, coords_br)
    vals_bl = tf.gather_nd(input, coords_bl)
    vals_tr = tf.gather_nd(input, coords_tr)

    h_offset = coords[:, 0] - tf.cast(coords_tl[:, 0], tf.float32)

    h_int_t = (((1.0 - h_offset) * vals_tl) + (h_offset * vals_tr))
    h_int_b = (((1.0 - h_offset) * vals_bl) + (h_offset * vals_br))

    v_offset = coords[:, 1] - tf.cast(coords_tl[:, 1], tf.float32)

    int_vals = (((1.0 - v_offset) * h_int_t) + (v_offset * h_int_b))

    return int_vals

def batch_map_coordinates(input, coords, n_coords):
    """Batch version of tf_map_coordinates"""
    #init_input_shape = input.get_shape()
    #input = tf.reshape(input, (-1, init_input_shape[3]))
    #coords = tf.reshape(coords, (-1, n_coords * 2))
    input_shape = input.get_shape()
    input_h = input_shape[1].value
    input_w = input_shape[2].value
    #batch_size = input_shape[0]
    #input_size = input_shape[1]

    #coords = tf.reshape(coords, (batch_size, -1, 2))

    #n_coords = tf.shape(coords)[1]

    coords_h = tf.clip_by_value(coords[..., :n_coords], 0, tf.cast(input_h, 'float32') - 1)
    coords_w = tf.clip_by_value(coords[..., n_coords:], 0, tf.cast(input_w, 'float32') - 1)
    coords = tf.stack([coords_h, coords_w], axis=-1)

    coords_tl = tf.cast(tf.floor(coords), 'float32')
    coords_br = tf.cast(tf.ceil(coords), 'float32')
    coords_bl = tf.stack([coords_tl[..., 0], coords_br[..., 1]], axis=-1)
    coords_tr = tf.stack([coords_br[..., 0], coords_tl[..., 1]], axis=-1)

    #idx = tf.range(batch_size)
    #idx = tf.expand_dims(idx, -1)
    #idx = tf.tile(idx, [1, n_coords])
    #idx = tf.reshape(idx, [-1])

    def _get_vals_by_coords(input, coords, n_coords):
        coords_shape = tf.shape(coords)
        input_shape = input.get_shape()
        input_w = input_shape[2].value
        input_h = input_shape[1].value
        channel_size = input_shape[3].value
        batch_size = tf.shape(input)[0]
        input = tf.transpose(input, (0, 3, 1, 2))
        input = tf.reshape(input, (-1, channel_size, input_h * input_w))

        indices = coords[..., 0] * input_w + coords[..., 1]
        #indices = tf.expand_dims(indices, axis=1)
        #indices = tf.tile(indices, [1, channel_size, 1, 1, 1])
        #indices = tf.reshape(indices, (-1, channel_size, input_h * input_w * n_coords))
        #indices = tf.transpose(indices, (0, 3, 1, 2))
        indices = tf.reshape(indices, (-1, input_h * input_w * n_coords))
        indices = tf.cast(indices, 'int32')
        #indices = tf.reshape(indices, [-1])
        #input = tf.reshape(input, [-1])
        vals = tf.gather(input, indices[0], axis=-1)
        #vals = tf.map_fn(lambda x: tf.gather(x[0], x[1], axis=-1), (input,indices), dtype=tf.float32)
        vals = tf.reshape(vals, (-1, channel_size, input_h, input_w, n_coords))
        return vals

    vals_tl = (1 + (coords_tl[..., 0] - coords[..., 0])) * \
       (1 + (coords_tl[..., 1] - coords[..., 1]))
    vals_br = (1 - (coords_br[..., 0] - coords[..., 0])) * \
       (1 - (coords_br[..., 1] - coords[..., 1]))
    vals_bl = (1 + (coords_bl[..., 0] - coords[..., 0])) * \
       (1 + (coords_bl[..., 1] - coords[..., 1]))
    vals_tr = (1 - (coords_tr[..., 0] - coords[..., 0])) * \
       (1 - (coords_tr[..., 1] - coords[..., 1]))

    x_vals_tl = _get_vals_by_coords(input, coords_tl, n_coords)
    x_vals_br = _get_vals_by_coords(input, coords_br, n_coords)
    x_vals_bl = _get_vals_by_coords(input, coords_bl, n_coords)
    x_vals_tr = _get_vals_by_coords(input, coords_tr, n_coords)

    #h_offset = coords[..., 0] - tf.cast(coords_tl[..., 0], tf.float32)

    #h_int_t = (((1.0 - h_offset) * vals_tl) + (h_offset * vals_tr))
    #h_int_b = (((1.0 - h_offset) * vals_bl) + (h_offset * vals_br))

    #v_offset = coords[..., 1] - tf.cast(coords_tl[..., 1], tf.float32)

    #int_vals = (((1.0 - v_offset) * h_int_t) + (v_offset * h_int_b))
    int_vals = tf.expand_dims(vals_tl, 1) * x_vals_tl + \
        tf.expand_dims(vals_br, 1) * x_vals_br + \
        tf.expand_dims(vals_bl, 1) * x_vals_bl + \
        tf.expand_dims(vals_tr, 1) * x_vals_tr
    return int_vals

def batch_map_offsets(input, offsets, order=1):
    """Batch map offsets into input
    Adds index of every entry to the entry to make it's interpolation
    relevant to it's location
    """

    offset_shape = offsets.get_shape()
    batch_size = tf.shape(offsets)[0]
    
    input_h = offset_shape[1]
    input_w = offset_shape[2]

    channel_size = int(offset_shape[3].value)
    #offsets = tf.reshape(offsets, (batch_size, -1, 2))
    #################### DEFAULT COORDINATES FOR EVERY POINT ####################
    ind_add = tf.meshgrid(
        tf.range(1, input_h + 1, delta=1), tf.range(1, input_w + 1, delta=1), indexing='ij'
    )
    ind_add = tf.stack(ind_add, axis=-1)
    ind_add = tf.cast(ind_add, 'float32')
    ind_add = tf.reshape(ind_add, (1, input_h, input_w, 2))
    ind_add = tf.tile(ind_add, [batch_size, 1, 1, int(channel_size / 2)])
    #############################################################################

    #################### KERNEL OFFSET FOR EVERY POINT ####################
    ind_zero = tf.meshgrid(
        tf.range(-1, 2, delta=1), tf.range(-1, 2, delta=1), indexing='ij'
    )
    ind_zero = tf.stack(ind_zero, axis=-1)
    ind_zero = tf.cast(ind_zero, 'float32')    
    ind_zero = tf.reshape(ind_zero, (1, 1, 1, channel_size))
    ind_zero = tf.tile(ind_zero, [batch_size, input_h, input_w, 1])
    #######################################################################

    coords = offsets + ind_add + ind_zero

    int_vals = batch_map_coordinates(input, coords, int(channel_size / 2))
    return int_vals