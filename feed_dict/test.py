import tensorflow as tf
a = tf.constant(0.1, dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32, shape=[])
c = a + b
with tf.Session() as s:
    tf.global_variables_initializer().run()
    print(s.run(c, feed_dict={c:1}))