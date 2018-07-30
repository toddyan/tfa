# -*- coding: utf-8 -*-
import globalconf
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
from image_process import ImageDataset

class Model:
    def __init__(self, num_classes):
        def _network_define(num_classes):
            with tf.variable_scope("input"):
                images = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3], name="input-image")
                labels = tf.placeholder(dtype=tf.int32, shape=[None], name="input-label")
            arg_scope = inception_v3.inception_v3_arg_scope()
            print("arg_scope:")
            for k, v in arg_scope.items():
                print(k)
                for k2, v2 in v.items():
                    print("  ", k2, v2)
            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                logits, _ = inception_v3.inception_v3(images, num_classes=num_classes)
            return images, labels, logits
        self.images, self.labels, logits = _network_define(num_classes)
        with tf.variable_scope("optimize"):
            tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=logits)
            self.train_op = tf.train.RMSPropOptimizer(0.00001).minimize(tf.losses.get_total_loss())
        with tf.variable_scope("pred"):
            self.predict = tf.argmax(logits, axis=1, output_type=tf.int32)
        with tf.variable_scope("eval"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
                self.labels,
                tf.argmax(logits, axis=1, output_type=tf.int32)
            ), tf.float32))

    def train(self, sess, x_train, y_train, batch_size, steps, in_model_path, out_model_path, logdir):
        def _get_restore_vars():
            restore_vars = []
            print("get_model_variables")
            for var in slim.get_model_variables():
                print("  ",var.op.name)
                if var.op.name.startswith('InceptionV3/Logits')\
                        or var.op.name.startswith('InceptionV3/AuxLogits'): continue
                restore_vars.append(var)
            return restore_vars
        sess.run(tf.global_variables_initializer())
        slim.assign_from_checkpoint_fn(in_model_path, _get_restore_vars(), ignore_missing_vars=True)(sess)
        # demonstrate
        print("trainable_variables:")
        for v in tf.trainable_variables():
            print("  ", v)
        writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
        writer.close()
        start = 0
        for step in range(steps):
            if start >= x_train.shape[0]: start = 0
            end = min(start + batch_size, x_train.shape[0])
            _, acc = s.run([self.train_op, self.accuracy],
                           feed_dict={self.images: x_train[start:end], self.labels: y_train[start:end]})
            print(step, acc)
            start = end
        saver = tf.train.Saver()
        saver.save(sess, out_model_path)

    def eval(self, sess, x_valid, y_valid, batch_size, model_path):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        start = 0
        while start < x_valid.shape[0]:
            end = min(start+batch_size, x_valid.shape[0])
            acc = sess.run(self.accuracy, feed_dict={self.images: x_valid[start:end], self.labels: y_valid[start:end]})
            print("valid:",acc)
            start = end

in_ckpt_path = globalconf.get_root() + "transfer/inception_v3/inception_v3.ckpt"
out_ckpt_path = globalconf.get_root() + "transfer/inception_v3/tuned_inception_v3.ckpt"
logdir = globalconf.get_root() + "transfer/inception_v3/logdir"

m = Model(5)
data = ImageDataset()
data.combine(globalconf.get_root() + 'transfer/small', 5, 0.1, 0.1, 0.5)
with tf.Session() as s:
    m.train(s,data.training_images, data.training_labels, 16, 10, in_ckpt_path, out_ckpt_path, logdir)

with tf.Session() as s:
    m.eval(s, data.validation_images, data.validation_labels, 64, out_ckpt_path)

