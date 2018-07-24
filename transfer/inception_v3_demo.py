# -*- coding: utf8 -*-
import globalconf
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

arg_scope = inception_v3.inception_v3_arg_scope()
print("arg_scope:")
for k, v in arg_scope.items():
    print(k)
    for k2, v2 in v.items():
        print("  ", k2, v2)

ckpt_path = globalconf.get_root() + "transfer/inception_v3/inception_v3.ckpt"
logdir = globalconf.get_root() + "transfer/inception_v3/logdir"
num_classes = 5
with tf.variable_scope("input"):
    images = tf.placeholder(dtype=tf.float32, shape=[None,299,299,3], name="input-image")
    labels = tf.placeholder(dtype=tf.int32, shape=[None], name="input-label")

logits, _ = inception_v3.inception_v3(images, num_classes=num_classes)

print("trainable_variables:")
for v in tf.trainable_variables():
    print("  ", v)
#writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
#writer.close()
