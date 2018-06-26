import numpy as np
import tensorflow as tf
import time

from fc import mnist_conf
from fc import infer

conf = mnist_conf.Conf()

with tf.variable_scope("input"):
    x = tf.placeholder(dtype=tf.float32, shape=[None, conf.fc_layers[0]], name="input-x")
    y = tf.placeholder(dtype=tf.int32, shape=[None], name="input-y")

logits = infer.infer(x, conf.fc_layers,None)
with tf.variable_scope("valid"):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(logits, axis=1, output_type=tf.int32),
        y
    ),tf.float32))

saver = tf.train.Saver()
with tf.Session() as s:
    tf.global_variables_initializer().run()
    while True:
        ckpt_stat = tf.train.get_checkpoint_state(conf.model_dir)
        if ckpt_stat and ckpt_stat.model_checkpoint_path:
            ckpt_path = ckpt_stat.model_checkpoint_path
            step = ckpt_path.split("/")[-1].split("-")[-1]
            saver.restore(s,ckpt_stat.model_checkpoint_path)
            print(step, ":", s.run(accuracy,feed_dict={x:conf.x_valid,y:conf.y_valid}))
        else:
            print("not check point file found.")
        time.sleep(3)