import numpy as np
import tensorflow as tf

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    #N, C, H, W = x_shape
    N = x_shape[0].value
    C = x_shape[1].value
    H = x_shape[2].value
    W = x_shape[3].value
    print(field_height.value)
    assert (H + 2 * padding - field_height.value) % stride == 0
    assert (W + 2 * padding - field_height.value) % stride == 0
    out_height = (H + 2 * padding - field_height.value) / stride + 1
    out_width = (W + 2 * padding - field_width.value) / stride + 1
    print(out_height, out_width)

    #i0 = tf_repeat(np.arange(field_height), field_width)
    i0 = np.repeat(np.arange(field_height.value), field_width.value)
    i0 = np.tile(i0, C)
    i0 = tf.convert_to_tensor(i0, dtype=tf.float32)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    i1 = tf.convert_to_tensor(i1, dtype = tf.float32)

    #j0 = tf.tile(tf.range(field_width.value), field_height * C)
    j0 = np.tile(np.arange(field_width.value), field_height.value * C)
    j0 = tf.convert_to_tensor(j0, dtype = tf.float32)
    j1 = stride * np.tile(np.arange(int(out_width)), int(out_height))
    j1 = tf.convert_to_tensor(j1, dtype = tf.float32)
    #i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    i = tf.reshape(i0,[-1,1]) + tf.reshape(i1,[1,-1])
    #j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    j = tf.reshape(j0,[-1,1])+tf.reshape(j1,[1,-1])

    #k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    #k = tf_repeat(np.arrange(C), field_height * field_width)
    k = np.repeat(np.arange(C), field_height.value * field_width.value)
    k = tf.convert_to_tensor(k, dtype = tf.float32)
    k = tf.reshape(k, [-1,1])

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    paddings = tf.constant([[0,0], [0,0], [p,p], [p,p]])
    x_padded = tf.pad(x, paddings, mode='constant')
    shape = x.get_shape()
    k, i, j = get_im2col_indices(shape, field_height, field_width, padding,
                               stride)
    cols = x_padded[:, k, i, j]
    #C = x.shape[1]
    C = shape[1].value
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
    x_padded = tf.sparse_add(x_padded, tf.SparseTensor(((slice(None), k, i, j), cols_reshaped)))

    if padding == 1:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
    repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
