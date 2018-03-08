import numpy as np
import tensorflow as tf
from im2col import get_im2col_indices
from im2col import im2col_indices
from im2col import col2im_indices

def dil(X, W, k, stride = 1, padding = 1):
    #n_filters, d_filter, h_filter, w_filter = tf.shape(W)
    #filter = tf.shape(W)
    filter = W.get_shape()
    #n_x, d_x, h_x, w_x = tf.shape(X)
    n = X.get_shape()
    h_out = (n[1].value - filter[0].value + 2 * padding) / stride + 1
    w_out = (n[2].value - filter[1].value + 2 * padding) / stride + 1

    #if not h_out.is_integer() or not w_out.is_integer():
    #    raise Exception('Invalid output dimension!')

    #h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, filter[0], filter[1], padding=padding, stride=stride)
    #W_col = W.reshape(n_filters, -1)
    W_col = tf.reshape(W, [filter[2].value, -1])

    out = tf.log(k* W_col @ exp(X_col))/k
    #out = out.reshape(n_filters, h_out, w_out, n_x)
    out = tf.reshape(out,[filter[0].value, h_out, w_out, n[0].value])
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

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [height, width])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    image = tf.expand_dims(image, -1)
    image = tf.image.resize_images(image, (28, 28))
    label = tf.one_hot(tf.cast(label, tf.int32), depth = 19)

    return image, label
