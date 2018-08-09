import numpy as np
import tensorflow as tf
import globalconf

def file_to_id_list(id_file):
    with open(id_file) as f:
        return list(map(
            lambda x:int(x),
            ' '.join([line.strip() for line in f.readlines()]).split(' ')
        ))


def make_batch(id_list, batch_size, timestep):
    num_batchs = (len(id_list)-1) // (batch_size * timestep)
    x = np.array(id_list[:num_batchs * batch_size * timestep])
    x = np.reshape(x, [batch_size, num_batchs * timestep])
    x = np.split(x, num_batchs, axis=1) # [num_batchs, batch_size, timestep]
    y = np.array(id_list[1:num_batchs * batch_size * timestep+1])
    y = np.reshape(y, [batch_size, num_batchs * timestep])
    y = np.split(y, num_batchs, axis=1) # [num_batchs, batch_size, timestep]
    return list(zip(x,y))


class Model(object):
    def __init__(self, batch_size, timestep, hidden_size, vocab_size):
        def network_define(timestep, hidden_size, max_grade_norm, learning_rate):
            with tf.variable_scope("cell"):
                cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for _ in range(2)
                ])
            with tf.variable_scope("input"):
                x = tf.placeholder(dtype=tf.int64, shape=[batch_size, timestep])
                y = tf.placeholder(dtype=tf.int64, shape=[batch_size, timestep])
            with tf.variable_scope("embeding", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                embeding = tf.get_variable("embeding", shape=[vocab_size, hidden_size], dtype=tf.float32)
            with tf.variable_scope("weight", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                weight = tf.get_variable("weight", shape=[hidden_size, vocab_size], dtype=tf.float32)
                bias = tf.get_variable("bias", shape=[vocab_size], dtype=tf.float32)
            self.embeded_x = tf.nn.embedding_lookup(embeding, x) # [batch_size, timestep, hidden_size]
            initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            state = initial_state
            outputs = [] # timestep * batch_size * hidden_size
            with tf.variable_scope("RNN"):
                for cur_timestep in range(timestep):
                    if cur_timestep > 0: tf.get_variable_scope().reuse_variables()
                    # shape of cell_output:          [batch_size, hidden_size]
                    # shape of state:      [layer, 2, batch_size, hidden_size]
                    cell_output, state = cell(self.embeded_x[:, cur_timestep, :], state)
                    outputs.append(cell_output)
            output = tf.concat(outputs, axis=1) #  batch_size, timestep * hidden_size
            output = tf.reshape(output, shape=[-1, hidden_size])
            logits = tf.nn.bias_add(tf.matmul(output, weight), bias)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(y, shape=[-1]),
                logits=logits
            )
            cost = tf.reduce_sum(loss) / batch_size
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(cost, tf.trainable_variables()),
                max_grade_norm
            )
            train_op = tf.train.GradientDescentOptimizer(learning_rate)\
                .apply_gradients(zip(grads, tf.trainable_variables()))
            return x, y, initial_state, state, train_op, cost
        self.x, self.y, self.initial_state, self.final_state, self.train_op, self.cost \
            = network_define(timestep, hidden_size, 5.0, 1.0)

    def run_epoch(self, data):
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()]
            state = sess.run(self.initial_state)
            for mini_x, mini_y in data:
                _, cost = s.run([self.train_op, self.cost])
if __name__ == "__main__":
    train_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/id.train.txt"
    r = file_to_id_list(train_file)
    print(r[0:1000])
    batch_size = 20
    timestep = 35
    data = make_batch(r, batch_size, timestep)
    for e in data:
        print(np.array(e).shape) #(2, 20, 35)
        exit()
