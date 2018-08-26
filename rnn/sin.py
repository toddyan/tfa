import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

TRAININT_EXAMPLES = 10000
TESTING_EXAMPLES  = 1000
SAMPLE_GAP = 0.01
TIMESTEP = 10
HIDDEN_SIZE = 30
test_end = (TRAININT_EXAMPLES + TESTING_EXAMPLES + 2 * TIMESTEP) * SAMPLE_GAP
train_seq = np.sin(np.linspace(0, (TRAININT_EXAMPLES + TIMESTEP) * SAMPLE_GAP, TRAININT_EXAMPLES + TIMESTEP).reshape([-1,1]))
test_seq = np.sin(np.linspace(
        (TRAININT_EXAMPLES + TIMESTEP) * SAMPLE_GAP,
        (TRAININT_EXAMPLES + TESTING_EXAMPLES + 2 * TIMESTEP) * SAMPLE_GAP,
        TESTING_EXAMPLES + TIMESTEP
    ).reshape([-1,1])
)

# plt.figure()
# plt.plot(seq[:1000])
# plt.legend()
# plt.show()
def make_dataset(seq):
    X = []
    y = []
    for i in range(len(seq)-TIMESTEP):
        X.append(seq[i:i+TIMESTEP])
        y.append(seq[i+TIMESTEP])
    X = np.array(X, np.float32)
    y = np.array(y, np.float32)
    return X, y
X_train, y_train = make_dataset(train_seq)
X_test, y_test = make_dataset(test_seq)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # (10000, 10, 1) (10000, 1) (1000, 10, 1) (1000, 1)

ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds = ds.repeat().shuffle(1000).batch(32)
X, y = ds.make_one_shot_iterator().get_next() # X:[batch,10,1], y:[batch,1]
cell = tf.nn.rnn_cell.MultiRNNCell([
    tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(2)
])
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)

rnn_output = rnn_outputs[:, -1, :]  # -1 for last time step (fixed sequence length for all examples)
# TODO try states.h instead
with tf.variable_scope("fc", initializer=tf.truncated_normal_initializer(stddev=0.1)):
    w = tf.get_variable("w", shape=[HIDDEN_SIZE, 1])
    b = tf.get_variable("b", shape=[1])
    predict = tf.matmul(rnn_output, w) + b
loss = tf.reduce_sum(tf.square(y - predict))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # a, b, o = sess.run([rnn_outputs, rnn_states, rnn_output])
    # print(a.shape, b[-1].c.shape, b[-1].h.shape, o.shape)
    for step in range(10000):
        _, lo = sess.run([train_op, loss])
        if step % 10 == 0: print(step, lo)
        if step % 10 == 0:
            pred = sess.run(predict, feed_dict={X:X_test})
            plt.figure()
            a = pred.reshape([-1])
            b = y_test.reshape([-1])
            plt.plot(a, label="predict")
            plt.plot(b, label="real")
            plt.legend()
            plt.show()
            print("testing loss:", sess.run(loss, feed_dict={X:X_test, y:y_test}))