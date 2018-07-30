# -*- coding: utf-8 -*-
import globalconf
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
from tf_record_dataset import ImageTFRecordBuilder
class Model:
    def __init__(self, num_classes):
        def _network_define(num_classes):
            with tf.variable_scope("input"):
                paser = ImageTFRecordBuilder(299, 299, 3, None, None, None, None, None, None).get_parser()
                # files = tf.train.match_filenames_once(train_files)
                tfrecord_paths = tf.placeholder(tf.string, shape=[])
                files = tf.train.match_filenames_once(tfrecord_paths)
                ds = tf.data.TFRecordDataset(files)
                ds = ds.map(paser).repeat(10).shuffle(buffer_size=100).batch(32)
                iterator = ds.make_initializable_iterator()
                images, labels = iterator.get_next()

            # with tf.variable_scope("input"):
            #     images = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3], name="input-image")
            #     labels = tf.placeholder(dtype=tf.int32, shape=[None], name="input-label")
            arg_scope = inception_v3.inception_v3_arg_scope()
            print("arg_scope:")
            for k, v in arg_scope.items():
                print(k)
                for k2, v2 in v.items():
                    print("  ", k2, v2)

            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                logits, _ = inception_v3.inception_v3(images, num_classes=num_classes)
            return images, labels, logits, iterator, tfrecord_paths
        self.images, self.labels, logits, self.iterator, self.tfrecord_paths = _network_define(num_classes)
        with tf.variable_scope("optimize"):
            tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=logits)
            self.train_op = tf.train.RMSPropOptimizer(0.00001).minimize(tf.losses.get_total_loss())
        with tf.variable_scope("pred"):
            self.predict = tf.argmax(logits, axis=1, output_type=tf.int64)
        with tf.variable_scope("eval"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
                self.labels,
                tf.argmax(logits, axis=1, output_type=tf.int64)
            ), tf.float32))
    def train(self, sess, in_model_path, out_model_path, logdir, train_files):
        def _get_restore_vars():
            restore_vars = []
            print("get_model_variables")
            for var in slim.get_model_variables():
                print("  ", var.op.name)
                if var.op.name.startswith('InceptionV3/Logits') \
                        or var.op.name.startswith('InceptionV3/AuxLogits'): continue
                restore_vars.append(var)
            return restore_vars
        sess.run(
            [tf.global_variables_initializer(), tf.local_variables_initializer()],
            feed_dict={self.tfrecord_paths: train_files}
        )
        sess.run(
            self.iterator.initializer,
            feed_dict={self.tfrecord_paths: train_files}
        )
        slim.assign_from_checkpoint_fn(in_model_path, _get_restore_vars(), ignore_missing_vars=True)(sess)
        # demonstrate
        print("trainable_variables:")
        for v in tf.trainable_variables():
            print("  ", v)
        writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
        writer.close()
        for step in range(256):
            try:
                _, acc = s.run([self.train_op, self.accuracy])
                print(step,acc)
            except tf.errors.OutOfRangeError:
                break
        saver = tf.train.Saver()
        saver.save(sess, out_model_path)

    def eval(self, sess, model_path, valid_files):
        sess.run(
            [tf.global_variables_initializer(), tf.local_variables_initializer()],
            feed_dict={self.tfrecord_paths: valid_files}
        )
        sess.run(
            self.iterator.initializer,
            feed_dict={self.tfrecord_paths: valid_files}
        )
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        while True:
            try:
                acc = sess.run(self.accuracy)
                print("valid:", acc)
            except tf.errors.OutOfRangeError:
                break
    def get_predictor(self, model_path):
        sess = tf.Session()
        sess.run(
            [tf.global_variables_initializer(), tf.local_variables_initializer()],
            feed_dict={self.tfrecord_paths: ''}
        )
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        def predictor(images):
            return sess.run(self.predict, feed_dict={self.images: images})
        return predictor

in_ckpt_path = globalconf.get_root() + "transfer/inception_v3/inception_v3.ckpt"
out_ckpt_path = globalconf.get_root() + "transfer/inception_v3/tuned_inception_v3.ckpt"
logdir = globalconf.get_root() + "transfer/inception_v3/logdir"
tfrecord_root = globalconf.get_root() + "transfer/tfrecord/"

m = Model(5)

# with tf.Session() as s:
#     m.train(s, in_ckpt_path, out_ckpt_path, logdir,tfrecord_root+"train-*")


# with tf.Session() as s:
#     m.eval(s, out_ckpt_path, tfrecord_root+"valid-*")
#     exit()

predictor = m.get_predictor(out_ckpt_path)
files = tf.train.match_filenames_once(tfrecord_root + "test-*")
paser = ImageTFRecordBuilder(299, 299, 3, None, None, None, None, None, None).get_parser()
ds = tf.data.TFRecordDataset(files)
ds = ds.map(paser).shuffle(buffer_size=128).batch(64)
iterator = ds.make_initializable_iterator()
image, label = iterator.get_next()
with tf.Session() as s:
    s.run([tf.global_variables_initializer(), tf.local_variables_initializer()], feed_dict={m.tfrecord_paths:''})
    s.run(iterator.initializer)
    for _ in range(10):
        try:
            img, lab = s.run([image, label])
            pred = predictor(img)
            acc = (pred == lab).astype(np.float32).mean()
            print(acc)
        except tf.errors.OutOfRangeError:
            break