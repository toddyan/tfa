import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

TRAININT_EXAMPLES = 10000
TESTING_EXAMPLES  = 1000
SAMPLE_GAP = 0.01
TIMESTEP = 10
test_end = (TRAININT_EXAMPLES + TESTING_EXAMPLES + 2 * TIMESTEP) * SAMPLE_GAP
seq = np.sin(np.linspace(0, (TRAININT_EXAMPLES + TIMESTEP) * SAMPLE_GAP, TRAININT_EXAMPLES + TIMESTEP).reshape([-1,1]))
# plt.figure()
# plt.plot(seq[:1000])
# plt.legend()
# plt.show()

X = []
y = []
for i in range(len(seq)-TIMESTEP):
    X.append(seq[i:i+TIMESTEP])
    y.append(seq[i+TIMESTEP])
X = np.array(X, np.float32)
y = np.array(y, np.float32)
print(X.shape, y.shape)  # (10000, 10, 1) (10000, 1)

ds = tf.data.Dataset.from_tensor_slices((X,y))
ds = ds.repeat().shuffle(1000).batch(4)
X, y = ds.make_one_shot_iterator().get_next() # X:[batch,10,1], y:[batch,1]
cell = tf.nn.rnn_cell.MultiRNNCell([
    tf.nn.rnn_cell.BasicLSTMCell(30) for _ in range(2)
])
outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    a, b = sess.run([outputs, states])
    print(a.shape, b[-1].c.shape, b[-1].h.shape)