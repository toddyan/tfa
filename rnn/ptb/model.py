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
    def __init__(self, batch_size, timestep, hidden_size, vocab_size, cell_dropout):
        def network_define(timestep, hidden_size, max_grade_norm, learning_rate, cell_dropout):
            with tf.variable_scope("cell"):
                cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size),output_keep_prob=cell_dropout)
                    for _ in range(2)
                ])
            with tf.variable_scope("input"):
                x = tf.placeholder(dtype=tf.int64, shape=[batch_size, timestep])
                y = tf.placeholder(dtype=tf.int64, shape=[batch_size, timestep])
            with tf.variable_scope("embeding", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                embeding = tf.get_variable("embeding", shape=[vocab_size, hidden_size], dtype=tf.float32)
                embeded_x = tf.nn.embedding_lookup(embeding, x)  # [batch_size, timestep, hidden_size]
            with tf.variable_scope("softmax", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                # weight = tf.get_variable("weight", shape=[hidden_size, vocab_size], dtype=tf.float32)
                weight = tf.transpose(embeding)
                bias = tf.get_variable("bias", shape=[vocab_size], dtype=tf.float32)
            with tf.variable_scope("initial_state"):
                initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            state = initial_state
            outputs = [] # timestep * batch_size * hidden_size
            with tf.variable_scope("RNN"):
                for cur_timestep in range(timestep):
                    if cur_timestep > 0: tf.get_variable_scope().reuse_variables()
                    # shape of cell_output:          [batch_size, hidden_size]
                    # shape of state:      [layer, 2, batch_size, hidden_size]
                    cell_output, state = cell(embeded_x[:, cur_timestep, :], state)
                    outputs.append(cell_output)
                output = tf.concat(outputs, axis=1) #  batch_size, timestep * hidden_size
                output = tf.reshape(output, shape=[-1, hidden_size])
            with tf.variable_scope("logits"):
                logits = tf.nn.bias_add(tf.matmul(output, weight), bias)
            with tf.variable_scope("optimize"):
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
            = network_define(timestep, hidden_size, 5.0, 1.0, cell_dropout)

    def train_epoch(self, data, sess):
        state = sess.run(self.initial_state)
        total_cost = 0.0
        iters = 0
        step = 0
        for mini_x, mini_y in data:
            _, cost, state = sess.run([self.train_op, self.cost, self.final_state],feed_dict={
                self.x: mini_x,
                self.y: mini_y,
                self.initial_state: state
            })
            timestep = mini_x.shape[1]
            total_cost += cost
            iters += timestep
            step += 1
            if step % 10 == 0:
                print("train step ", step, ", perplexity:", np.exp(total_cost/iters))

    def validate(self, data, sess):
        state = sess.run(self.initial_state)
        total_cost = 0.0
        iters = 0
        step = 0
        for mini_x, mini_y in data:
            cost, state = sess.run([self.cost, self.final_state], feed_dict={
                self.x: mini_x,
                self.y: mini_y,
                self.initial_state: state
            })
            timestep = mini_x.shape[1]
            total_cost += cost
            iters += timestep
            step += 1
            if step % 10 == 0:
                print("valid step ", step, ", perplexity:", np.exp(total_cost / iters))


def train():
    train_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/id.train.txt"
    model_path = globalconf.get_root() + "rnn/ptb/model/model.ckpt"
    train_id_list = file_to_id_list(train_file)
    batch_size = 20
    timestep = 35
    hidden_size = 300
    vocab_size = 10000
    data = make_batch(train_id_list, batch_size, timestep)
    model = Model(batch_size, timestep, hidden_size, vocab_size, 0.9)
    writer = tf.summary.FileWriter(logdir=globalconf.get_root()+"rnn/ptb/log", graph=tf.get_default_graph())
    writer.close()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        for epoch in range(10):
            print("epoch ", epoch)
            model.train_epoch(data, sess)
            saver = tf.train.Saver()
            saver.save(sess, save_path=model_path)

def valid():
    valid_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/id.valid.txt"
    model_path = globalconf.get_root() + "rnn/ptb/model/model.ckpt"
    valid_id_list = file_to_id_list(valid_file)
    batch_size = 20
    timestep = 35
    hidden_size = 300
    vocab_size = 10000
    valid_data = make_batch(valid_id_list, batch_size, timestep)
    model = Model(batch_size, timestep, hidden_size, vocab_size, 1.0)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver.restore(sess, model_path)
        model.validate(valid_data, sess)

if __name__ == "__main__":
    train()
    g = tf.Graph()
    with g.as_default():
        valid()
