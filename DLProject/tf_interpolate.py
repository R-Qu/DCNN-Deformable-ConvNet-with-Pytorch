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

def batch_map_coordinates(input, coords, order=1):
    """Batch version of tf_map_coordinates"""
    
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]

    #coords = tf.reshape(coords, (batch_size, -1, 2))

    n_coords = tf.shape(coords)[1]

    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)

    coords_tl = tf.cast(tf.floor(coords), 'int32')
    coords_br = tf.cast(tf.ceil(coords), 'int32')
    coords_bl = tf.stack([coords_tl[..., 0], coords_br[..., 1]], axis=-1)
    coords_tr = tf.stack([coords_br[..., 0], coords_tl[..., 1]], axis=-1)

    idx = tf.range(batch_size)
    idx = tf.expand_dims(idx, -1)
    idx = tf.tile(idx, [1, n_coords])
    idx = tf.reshape(idx, [-1])

    def _get_vals_by_coords(input, coords):
        coords_0_flat = tf.reshape(coords[..., 0], [-1])
        coords_1_flat = tf.reshape(coords[..., 1], [-1])
        indices = tf.stack([idx, coords_0_flat, coords_1_flat], axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords))
        return vals

    vals_tl = _get_vals_by_coords(input, coords_tl)
    vals_br = _get_vals_by_coords(input, coords_br)
    vals_bl = _get_vals_by_coords(input, coords_bl)
    vals_tr = _get_vals_by_coords(input, coords_tr)

    h_offset = coords[..., 0] - tf.cast(coords_tl[..., 0], tf.float32)

    h_int_t = (((1.0 - h_offset) * vals_tl) + (h_offset * vals_tr))
    h_int_b = (((1.0 - h_offset) * vals_bl) + (h_offset * vals_br))

    v_offset = coords[..., 1] - tf.cast(coords_tl[..., 1], tf.float32)

    int_vals = (((1.0 - v_offset) * h_int_t) + (v_offset * h_int_b))

    return int_vals

def batch_map_offsets(input, offsets, order=1):
    """Batch map offsets into input
    Adds index of every entry to the entry to make it's interpolation
    relevant to it's location
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_w = input_shape[1]
    input_h = input_shape[2]
    offsets = tf.reshape(offsets, (batch_size, -1, 2))

    ind_add = tf.meshgrid(
        tf.range(input_w), tf.range(input_h), indexing='ij'
    )
    ind_add = tf.stack(ind_add, axis=-1)
    ind_add = tf.cast(ind_add, 'float32')
    ind_add = tf.reshape(ind_add, (-1, 2))
    ind_add = tf.expand_dims(ind_add, 0)
    ind_add = tf.tile(ind_add, [batch_size, 1, 1])

    coords = offsets + ind_add

    int_vals = batch_map_coordinates(input, coords)
    return int_vals