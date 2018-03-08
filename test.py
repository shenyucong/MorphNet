import tensorflow as tf
from tool import dil
import os
import tool

def monet(x):
    w = tool.weight_variable([3, 3, 1, 32])
    y_mo = dil(x, w, 10)

    return y_mo

tfrecords_file_train = 'bees_train.tfrecords'
train_dir = '/Users/chenyucong/Desktop/research/ecology/'
filename = os.path.join(train_dir, tfrecords_file_train)
with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])
    images, label = tool.read_and_decode(filename_queue)
    image_batch, label_batch = tf.train.shuffle_batch([images, label], batch_size = 25, num_threads = 1, capacity = 1000+3*25, min_after_dequeue = 1000)


x = tf.placeholder(tf.int32, [None, 28, 28, 1], name = "x1")

y = monet(x)

with tf.session() as sess:
    x1 = sess.run([image_batch])
    print(y.eval(feed_dict={x: x1}))
