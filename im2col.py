import numpy as np
import tensorflow as tf

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = tf.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = tf.tile(tf.arange(field_width), field_height * C)
    j1 = stride * tf.tile(tf.arange(out_width), out_height)
    #i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    i = tf.reshape(i0,[-1,1]) + tf.reshape(i1,[1,-1])
    #j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    j = tf.reshape(j0,[-1,1])+tf.reshape(j1,[1,-1])

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    paddings = tf.constant([[0,0], [0,0], [p,p], [p,p]])
    x_padded = tf.pad(x, paddings, mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)
    cols = x_padded[:, k, i, j]
    #C = x.shape[1]
    C = tf.shape(x)[1]
    #cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    cols = tf.transpose(cols, perm=[1,2,0])
    cols = tf.reshape(cols, [field_height * field_width *C, -1])
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = tf.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
    cols_reshaped = tf.reshape(cols, [C * field_height * field_width, -1, N])
    cols_reshaped = tf.transpose(cols_reshaped,perm =[2,0,1])
    #np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    tf.sparse_add(x_padded, tf.SparseTensor(x, tf.reshape(dout, [-1])))
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

pass
