import tensorflow as tf


class Model:
    def __init__(self, h, w, c, conv_layers, dense_layers):

        def _netword_define(h,w,c,conv_layers,dense_layers):
            initializer = tf.truncated_normal_initializer(stddev=0.1)
            with tf.variable_scope("input"):
                x = tf.placeholder(dtype=tf.float32, shape=[None, h, w, c])
                y = tf.placeholder(dtype=tf.int32, shape=[None])
            a = x
            prev_c = [c]
            for i in range(len(conv_layers)):
                layer_name = "conv_layer" + str(i+1)
                layer = conv_layers[i]
                with tf.variable_scope(layer_name, initializer=initializer):
                    filter_shape = layer[0] + prev_c + layer[1]
                    pool_size = [1] + layer[2] + [1]
                    w = tf.get_variable("w", shape=filter_shape, dtype=tf.float32)
                    b = tf.get_variable("b", shape=layer[1], dtype=tf.float32)
                    a = tf.nn.conv2d(a, filter=w, strides=[1,1,1,1],padding='SAME')
                    a = tf.nn.relu(tf.nn.bias_add(a, b))
                    a = tf.nn.max_pool(a, ksize=pool_size, strides=pool_size, padding='SAME')
                    prev_c = layer[1]
            with tf.variable_scope("flatten", initializer=initializer):
                conv_out_shape = a.get_shape().as_list()
                flatten_size = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]
                a = tf.reshape(a, shape=[-1, flatten_size])
            prev_size = flatten_size
            for i in range(len(dense_layers)):
                layer_name = "dense" + str(i+1)
                cur_size = dense_layers[i]
                with tf.variable_scope(layer_name, initializer=initializer):
                    w = tf.get_variable("w", shape=[prev_size, cur_size], dtype=tf.float32)
                    b = tf.get_variable("b", shape=[cur_size], dtype=tf.float32)
                    logits = tf.matmul(a, w) + b
                    if i + 1 != len(dense_layers):
                        a = tf.nn.relu(logits)
                prev_size = cur_size
            return x,y,logits

        self.x,self.y,logits = _netword_define(h,w,c,conv_layers,dense_layers)
        with tf.variable_scope("optimizer"):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=logits))
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
            self.predict = tf.argmax(logits, axis=1, output_type=tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.predict), tf.float32))
    def train(self, sess, x_train, y_train, batch_size, steps, model_path):
        start = 0
        for step in range(steps):
            if start >= x_train.shape[0]: start = 0
            end = min(start + batch_size, x_train.shape[0])
            _, acc = sess.run(
                [self.train_op, self.accuracy],
                feed_dict={self.x:x_train[start:end],self.y:y_train[start:end]}
            )
            if step % 10 == 0: print(step, acc)
            start = end
        saver = tf.train.Saver()
        saver.save(sess, model_path)
    def eval(self, sess, x_valid, y_valid, batch_size, model_path):
        saver = tf.train.Saver() # no name map
        saver.restore(sess, model_path)
        correct_predict = tf.reduce_sum(tf.cast(tf.equal(self.y, self.predict),tf.float32))
        sum = 0.0
        start = 0
        while start < x_valid.shape[0]:
            end = min(start+batch_size, x_valid.shape[0])
            sum += sess.run(correct_predict, feed_dict={self.x:x_valid[start:end], self.y:y_valid[start:end]})
            start = end
        return sum/x_valid.shape[0]


conv_layers = [[[5,5],[16],[2,2]],[[5,5],[32],[2,2]]]
dense_layers = [512, 128, 10]
from keras.datasets import cifar10
import numpy as np
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.reshape([-1,32,32,3])/255.0
x_test = x_test.reshape([-1,32,32,3])/255.0
y_train = y_train.reshape(-1).astype(np.int32)
y_test = y_test.reshape(-1).astype(np.int32)
a = Model(32,32,3, conv_layers, dense_layers)
import globalconf
model_dir = globalconf.get_root() + "/one_cnn/model/"
model_name = "model.ckpt"
with tf.Session() as s:
    tf.global_variables_initializer().run()
    a.train(s,x_train,y_train,64,100, model_dir+model_name)

with tf.Session() as s:
    tf.global_variables_initializer().run()
    print(a.eval(s,x_test,y_test,128,model_dir+model_name))