import tensorflow as tf
import numpy as np
import random

from keras.datasets import boston_housing
(x_train, y_train),(x_test, y_test) = boston_housing.load_data()
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train-mean)/std
x_test = (x_test-mean)/std

feature = tf.placeholder(dtype=tf.float32, shape=[None, 13], name="feature")
price   = tf.placeholder(dtype=tf.float32, shape=[None],     name="price")
u = tf.get_variable("weight", shape=[13,1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
W = tf.get_variable("sqr_weight", shape=[13,13], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
b = tf.get_variable("bias", shape=[], dtype=tf.float32, initializer=tf.constant_initializer())
sqr_part = tf.reshape(tf.diag_part(tf.matmul(tf.matmul(feature, W), tf.transpose(feature))),[-1,1])
pred = b + tf.matmul(feature, u)  + sqr_part
loss = tf.reduce_mean(tf.square(price-pred))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
start = 0
batch_size = 32
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(1000000):
        if start >= x_train.shape[0]:
            start = 0
        end = min(x_train.shape[0], start + batch_size)
        sess.run(train_op, feed_dict={feature:x_train[start:end], price:y_train[start:end]})
        start = end
        if step % 1000 == 0:
            print(step, sess.run(loss, feed_dict={feature:x_test, price:y_test}))

# sqr lr: learning_rate 0.001, batch_size 32 test_loss 85.48

