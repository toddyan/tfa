import numpy as np
import tensorflow as tf
timestep = 2
input_dim = 3
hidden_size = 5

x = tf.placeholder(dtype=tf.float32, shape=[None, timestep, input_dim])
seq_lenth = tf.placeholder(dtype=tf.int64, shape=[None])

# [batch_size, timestep, hidden_size]
x_data = np.array([
  # t = 0      t = 1
  [[0, 1, 2], [9, 8, 7]],  # instance 0
  [[3, 4, 5], [0, 0, 0]],  # instance 1
  [[6, 7, 8], [6, 5, 4]],  # instance 2
  [[9, 0, 1], [3, 2, 1]],  # instance 3
])

# cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
# cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(6)])
# cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for _ in range(1)])
initial_state = cell.zero_state(batch_size=4, dtype=tf.float32)


outputs, states = tf.nn.dynamic_rnn(cell=cell,inputs=x,sequence_length=seq_lenth, dtype=tf.float32)
output, state = cell(tf.constant(x_data[:, 0, :], dtype=tf.float32), initial_state)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    o, s = sess.run([outputs, states], feed_dict={x:x_data, seq_lenth:[2,1,2,2]})

    # single layer, basicRNN:    [batch_size, timestep, hidden_size]
    # multiple layer, basicRNN:  [batch_size, timestep, hidden_size], last layer
    # single layer, basicLSTM:   [batch_size, timestep, hidden_size]
    # multiple layer, basicLSTM: [batch_size, timestep, hidden_size], last layer
    print(np.array(o).shape)
    print(o)
    print("----")

    # single layer, basicRNN:              [batch_size, hidden_size]
    # multiple layer, basicRNN:  [layer,    batch_size, hidden_size]
    # single layer, basicLSTM:          [2, batch_size, hidden_size], where "2" for c and h of LSTM
    # multiple layer, basicLSTM: [layer, 2, batch_size, hidden_size]
    print(np.array(s).shape)
    print(s)

    print("--- single cell ---")
    o, s = sess.run([output, state])

    # single layer, basicRNN:    [batch_size, hidden_size]
    # multiple layer, basicRNN:  [batch_size, hidden_size]
    # single layer, basicLSTM:   [batch_size, hidden_size]
    # multiple layer, basicLSTM: [batch_size, hidden_size]
    print(np.array(o).shape)
    print(o)
    print("----")

    # single layer, basicRNN:              [batch_size, hidden_size]
    # multiple layer, basicRNN:  [layer,    batch_size, hidden_size]
    # single layer, basicLSTM:          [2, batch_size, hidden_size]
    # multiple layer, basicLSTM: [layer, 2, batch_size, hidden_size]
    print(np.array(s).shape)
    print(s)