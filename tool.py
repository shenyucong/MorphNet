import numpy as np
import tensorflow as tf

def dil(k,x,W):
    '''k: smoothing parameters
       x: input tensor in four dimension (in_channels, width, heights, out_channels)
       W: structuing element (height, width, in_channels, out_channels)'''
    shape_input = x.get_shape()
    shape_weights = W.get_shape()
    shape_output = np.array([shape_input[0], shape_input[1]-shap_weights[0]+1, shape_input[2]-shape_weights+1, shape_input[3]])
    output = tf.zeros(shape_input, tf.int32)
    for i = (shape_weights[0]-1)/2-1:shape_input[1]-(shape_weights[0]-1)/2-1:
        for j = (shape_weights[0]-1)/2-1:shape_input[2]-(shape_weights[0]-1)/2-1:
            sigma = 0
            for a = 0:shape_weights[0]-1:
                for b = 0:shape_weights[1]-1:
                    sigma += W[a,b,-1,-1]*tf.exp(x[-1,i+a,j+b,-1])
            output[]-1, i, j, -1] = tf.log(k*sigma)/k

    output_image = output[-1,(shape_weights[0]-1)/2-1:shape_input[1]-(shape_weights[0]-1)/2-1,
                          (shape_weights[0]-1)/2-1:ahpe_input[1]-(shape_weights[0]-1)/2-1.-1]
    return output_image

def ero(k,x,W):
'''k: smoothing parameters
       x: input tensor in four dimension (in_channels, width, heights, out_channels)
       W: structuing element (height, width, in_channels, out_channels)'''
    shape_input = x.get_shape()
    shape_weights = W.get_shape()
    shape_output = np.array([shape_input[0], shape_input[1]-shap_weights[0]+1, shape_input[2]-shape_weights+1, shape_input[3]])
    output = tf.zeros(shape_input, tf.int32)
    for i = (shape_weights[0]-1)/2-1:shape_input[1]-(shape_weights[0]-1)/2-1:
        for j = (shape_weights[0]-1)/2-1:shape_input[2]-(shape_weights[0]-1)/2-1:
            sigma = 0
            for a = 0:shape_weights[0]-1:
                for b = 0:shape_weights[1]-1:
                    sigma += W[a,b,-1,-1]*tf.exp(x[-1,i+a,j+b,-1])
            output[]-1, i, j, -1] = -tf.log(-k*sigma)/k

    output_image = output[-1,(shape_weights[0]-1)/2-1:shape_input[1]-(shape_weights[0]-1)/2-1,
                          (shape_weights[0]-1)/2-1:ahpe_input[1]-(shape_weights[0]-1)/2-1.-1]
    return output_image



def weight_variable(shape):
    '''weight_variable generates a weight variable of a given shape.'''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = "W")

def bias_variable(shape):
    '''bias_variable generates a bias variable of a given shape'''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = "B")
