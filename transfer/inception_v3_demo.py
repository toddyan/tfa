# -*- coding: utf-8 -*-
import globalconf
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
from image_process import ImageDataset

def get_restore_vars():
    restore_vars = []
    print("get_model_variables")
    for var in slim.get_model_variables():
        print("  ",var.op.name)
        if var.op.name.startswith('InceptionV3/Logits')\
                or var.op.name.startswith('InceptionV3/AuxLogits'): continue
        restore_vars.append(var)
    return restore_vars

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

with tf.variable_scope("optimize"):
    tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    train_op = tf.train.RMSPropOptimizer(0.00001).minimize(tf.losses.get_total_loss())

with tf.variable_scope("eval"):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        labels,
        tf.argmax(logits, axis=1, output_type=tf.int32)
    ),tf.float32))

restore_fn = slim.assign_from_checkpoint_fn(ckpt_path,get_restore_vars(),ignore_missing_vars=True)

# demonstrate
print("trainable_variables:")
for v in tf.trainable_variables():
    print("  ", v)
writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
writer.close()

with tf.Session() as s:
    tf.global_variables_initializer().run()
    restore_fn(s)
    data = ImageDataset()
    data.combine(globalconf.get_root() + 'transfer/small', 5, 0.1, 0.1, 0.5)
    start = 0
    batch_size = 32
    for step in range(1000):
        if start >= data.training_images.shape[0]: start = 0
        end = min(start+batch_size, data.training_images.shape[0])
        _, acc = s.run([train_op, accuracy], feed_dict={images:data.training_images[start:end], labels:data.training_labels[start:end]})
        print(step,acc)
        start = end


