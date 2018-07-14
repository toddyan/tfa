from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import globalconf

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)/255.0
x_test = x_test.reshape(-1,28,28,1)/255.0
print(x_train.shape,y_train.shape)

with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="input-x")
    y = tf.placeholder(tf.int32, shape=[None], name="input-y")

initializer = tf.truncated_normal_initializer(stddev=0.1)
with tf.variable_scope("layer1", initializer=initializer):
    w = tf.get_variable("w", shape=[28,28,1,16])
    b = tf.get_variable("b", shape=[16])
    a = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
    a = tf.nn.relu(tf.nn.bias_add(a, b))
    a = tf.nn.max_pool(a, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

with tf.variable_scope("layer2", initializer=initializer):
    w = tf.get_variable("w", shape=[28,28,16,32])
    b = tf.get_variable("b", shape=[32])
    a = tf.nn.conv2d(a, w, strides=[1,1,1,1], padding='SAME')
    a = tf.nn.relu(tf.nn.bias_add(a, b))
    a = tf.nn.max_pool(a, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

with tf.variable_scope("flatten", initializer=initializer):
    conv_out_shape = a.get_shape().as_list()
    flatten_size = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]
    a = tf.reshape(a, shape=[-1, flatten_size])

with tf.variable_scope("layer3", initializer=initializer):
    w = tf.get_variable("w", shape=[flatten_size,512], dtype=tf.float32)
    b = tf.get_variable("b", shape=[512], dtype=tf.float32)
    a = tf.nn.relu(tf.matmul(a, w) + b)

with tf.variable_scope("layer4", initializer=initializer):
    w = tf.get_variable("w", shape=[512,128], dtype=tf.float32)
    b = tf.get_variable("b", shape=[128], dtype=tf.float32)
    a = tf.nn.relu(tf.matmul(a, w) + b)

with tf.variable_scope("layer5", initializer=initializer):
    w = tf.get_variable("w", shape=[128,10], dtype=tf.float32)
    b = tf.get_variable("b", shape=[10], dtype=tf.float32)
    logits = tf.matmul(a, w) + b

with tf.variable_scope("optimize", initializer=initializer):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(logits, axis=1,output_type=tf.int32), y
    ),tf.float32))

    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

log_dir = globalconf.get_root() + "one_cnn/straight_foeward"
writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())
writer.close()

with tf.Session() as s:
    tf.global_variables_initializer().run()
    for epoch in range(10):
        start = 0
        batch_size = 64
        while start < x_train.shape[0]:
            end = min(x_train.shape[0], start + batch_size)
            lo, acc, _ = s.run([loss, accuracy, train_op], feed_dict={x:x_train[start:end],y:y_train[start:end]})
            print(lo,acc)
            start = end
            if start/64%10==0:
                acc = s.run(accuracy, feed_dict={x:x_test[0:3000],y:y_test[0:3000]})
                print(epoch, acc)
