import tensorflow as tf
import tensorflow.contrib.layers as cont_layers

from fc import cifar10_conf
from fc import infer
conf = cifar10_conf.Conf()
regul = cont_layers.l2_regularizer(conf.regularizer_weight)
with tf.variable_scope("input"):
    x = tf.placeholder(dtype=tf.float32, shape=[None, conf.fc_layers[0]], name="input-x")
    y = tf.placeholder(dtype=tf.int32, shape=[None], name="input-y")
logits = infer.infer(x, conf.fc_layers, regul)
global_step = tf.get_variable("global_step", shape=[], initializer=tf.zeros_initializer(), trainable=False)
with tf.variable_scope("optimizer"):
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y,
        logits=logits
    ))
    total_loss = cross_entropy + tf.add_n(tf.get_collection("losses"))
    print("shape",conf.x_train.shape[0])
    learning_rate = tf.train.exponential_decay(
        learning_rate=conf.learning_rate_base,
        global_step = global_step,
        decay_steps=int(conf.x_train.shape[0]/conf.batch_size),
        decay_rate=conf.learning_rate_decay,
        staircase=True
    )
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss,global_step=global_step)
with tf.variable_scope("ema"):
    ema = tf.train.ExponentialMovingAverage(
        decay=conf.ema_decay,
        num_updates=global_step
    )
    ema_op = ema.apply(tf.trainable_variables())
group_op = tf.group(train_op,ema_op)
writer = tf.summary.FileWriter(conf.model_dir, tf.get_default_graph())
writer.close()
saver = tf.train.Saver()
with tf.Session() as s:
    tf.global_variables_initializer().run()
    while True:
        x_batch, y_batch = conf.get_batch()
        step, lo, ce, _ = s.run([global_step, total_loss, cross_entropy, group_op], feed_dict={x:x_batch,y:y_batch})
        if step%10 == 0: print(step,":",lo,ce)
        if step%100 == 0:
            saver.save(s,conf.model_path,global_step=global_step)
        if step==50000:
            break