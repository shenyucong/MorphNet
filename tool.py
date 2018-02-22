import numpy as np
import tensorflow as tf
from im2col import get_im2col_indices
from im2col import im2col_indices
from im2col import col2im_indices

def dil(X, W, k, stride = 1, padding = 1):
    #n_filters, d_filter, h_filter, w_filter = tf.shape(W)
    filter = tf.shape(W)
    #n_x, d_x, h_x, w_x = tf.shape(X)
    n = tf.shape(X)
    h_out = (n[2] - filter[2] + 2 * padding) / stride + 1
    w_out = (n[3] - filter[3] + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, filter[2], filter[3], padding=padding, stride=stride)
    #W_col = W.reshape(n_filters, -1)
    W_col = tf.reshape(W, [filter[0], -1])

    out = tf.log(k* W_col @ exp(X_col))/k
    #out = out.reshape(n_filters, h_out, w_out, n_x)
    out = tf.reshape(out,[filter[0], h_out, w_out, n[0]])
    #out = out.transpose(3, 0, 1, 2)
    out = tf.transpose(out, perm = [3, 0, 1, 2])

    #cache = (X, W, b, stride, padding, X_col)

    return out

def weight_variable(shape):
    '''weight_variable generates a weight variable of a given shape.'''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = "W")

def bias_variable(shape):
    '''bias_variable generates a bias variable of a given shape'''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = "B")
