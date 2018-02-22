import tensorflow as tf
from tool import dil

x = tf.constant([[1,2,3,4],[2,3,4,5],[3,4,5,7],[3,5,7,9]])
w = tf.constant([[0,0,1],[2,3,5],[3,4,6]])
y = dil(x, w, 10)

x1 = tf.placeholder(tf.int32, [4,4], name = "x1")

with tf.session() as sess:
    x = sess.run([x])
    print(y.eval(feed_dict={x1: x}))
