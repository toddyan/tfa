import numpy as np
import tensorflow as tf
import globalconf
final_en_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.en.final"
final_zh_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.zh.final"

def file_to_dataset(file):
    dataset = tf.data.TextLineDataset(file)
    # input Shape must be rank 1 for 'StringSplit', output shape is rank 1
    return dataset.map(lambda line: tf.string_split([line]).values)\
            .map(lambda str_tokens: tf.cast(tf.string_to_number(string_tensor=str_tokens), tf.int64))

def join_src_trg_dataset(src_dataset, trg_dataset, batch_size, trg_sos_id, trg_eos_id):
    # rank 2: ([src list],[trg list])
    dataset = tf.data.Dataset.zip((src_dataset, trg_dataset))
    def extend_trg(src_list, trg_list):
        # rank 1
        trg_in = tf.concat([[trg_sos_id], trg_list], axis=0)
        trg_out = tf.concat([trg_list, [trg_eos_id]], axis=0)
        return (src_list, tf.size(src_list), trg_in, trg_out, tf.size(trg_out))
    dataset = dataset.map(extend_trg).shuffle(10000).padded_batch(batch_size, padded_shapes=(
        (tf.TensorShape([None]), tf.TensorShape([]),
            tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]))
    ))
    iter = dataset.make_one_shot_iterator()
    return iter.get_next()

class Model(object):
    def __init__(self, src_dataset, trg_dataset, batch_size, trg_sos_id, trg_eos_id, src_vocab_size, trg_vocab_size, hidden_size, layers):
        def network_define(src_dataset, trg_dataset, batch_size, trg_sos_id, trg_eos_id, src_vocab_size, trg_vocab_size, hidden_size, layers):
            with tf.variable_scope("input"):
                src_seq, src_size, trg_in_seq, trg_out_seq, trg_size\
                    = join_src_trg_dataset(src_dataset, trg_dataset, batch_size, trg_sos_id, trg_eos_id)
            with tf.variable_scope("embding", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                enc_embeding = tf.get_variable("src_embeding", shape=[src_vocab_size, hidden_size])
                dec_embeding = tf.get_variable("trg_embeding", shape=[trg_vocab_size, hidden_size])
                src_embeded = tf.nn.embedding_lookup(enc_embeding, src_seq)
                trg_in_embeded = tf.nn.embedding_lookup(dec_embeding, trg_in_seq)
            with tf.variable_scope("cells", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                enc_cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for _ in range(layers)
                ])
                dec_cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for _ in range(layers)
                ])

            with tf.variable_scope("encoder", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                _, enc_state = tf.nn.dynamic_rnn(enc_cell, src_embeded, src_size, dtype=tf.float32)
            with tf.variable_scope("decoder", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                dec_output, _ = tf.nn.dynamic_rnn(dec_cell, trg_in_embeded, trg_size, initial_state=enc_state)
            with tf.variable_scope("softmax", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                w_softmax = tf.transpose(dec_embeding)
                b_softmax = tf.get_variable("bias_softmax", shape=[trg_vocab_size], dtype=tf.float32)

            with tf.variable_scope("loss", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                logits = tf.nn.bias_add(tf.matmul(tf.reshape(dec_output, shape=[-1, hidden_size]), w_softmax), b_softmax)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.reshape(trg_out_seq, shape=[-1]),
                    logits=tf.reshape(logits, shape=[-1, trg_vocab_size])
                )
                mask = tf.reshape(tf.sequence_mask(trg_size, maxlen=tf.shape(trg_in_seq)[1], dtype=tf.float32), [-1])
                cost = loss * mask
                token_cost = cost / tf.reduce_sum(mask)
            with tf.variable_scope("optimize", initializer=tf.truncated_normal_initializer(stddev=0.1)):
                grads = tf.gradients(cost/tf.to_float(batch_size), tf.trainable_variables())
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0) # TODO
                train_op = tf.train.GradientDescentOptimizer(1.0).apply_gradients(zip(grads, tf.trainable_variables()))
            return src_seq, src_size, trg_in_seq, trg_out_seq, trg_size, enc_cell, dec_cell, token_cost, train_op
        src_seq, src_size, trg_in_seq, trg_out_seq, trg_size, enc_cell, dec_cell, self.token_cost, self.train_op = \
            network_define(src_dataset, trg_dataset, batch_size, trg_sos_id, trg_eos_id, src_vocab_size, trg_vocab_size, hidden_size, layers)
    def train(self):
        step = 0
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            for epoch in range(5):
                while True:
                    try:
                        co, _ = sess.run([tf.reduce_sum(self.token_cost), self.train_op])
                        step += 1
                        print(co)
                    except tf.errors.OutOfRangeError: break

if __name__ == "__main__":
    src_ds = file_to_dataset(final_en_file)
    trg_ds = file_to_dataset(final_zh_file)
    model = Model(src_ds, trg_ds, 100, 0, 1, 10000, 10000, 1024, 2)
    model.train()

if __name__ == "__main__1":
    src_ds = file_to_dataset(final_en_file)
    trg_ds = file_to_dataset(final_zh_file)
    ds = join_src_trg_dataset(src_ds, trg_ds, 2, 0, 1)
    iter = ds.make_one_shot_iterator()
    line = iter.get_next()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        for _ in range(1):
            print(sess.run(line))
