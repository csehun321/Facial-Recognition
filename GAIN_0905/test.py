import tensorflow as tf
import numpy as np

tfZeros = tf.ones([64,1], tf.int32)
test= tf.ones([16,16,64], tf.int32)

tfreshape = tf.reshape(tfZeros, (-1,-1,1))
k = tf.matmul(test,tfreshape)
print(test, tfreshape)
with tf.Session() as sess:
    a = sess.run(k)
    print(a)