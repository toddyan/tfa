import numpy as np
import tensorflow as tf
import codecs
import globalconf
from tensorflow.contrib import seq2seq
def file_to_dataset(file_holder):
    files = tf.train.match_filenames_once(file_holder)
    dataset = tf.data.TextLineDataset(files)
    # input Shape must be rank 1 for 'StringSplit', output shape is rank 1
    return dataset.map(lambda line: tf.string_split([line]).values)\
            .map(lambda str_tokens: tf.cast(tf.string_to_number(string_tensor=str_tokens), tf.int64))

def join_src_trg_dataset(src_path, trg_path, batch_size, trg_sos_id, trg_eos_id):
    src_dataset = file_to_dataset(src_path)
    trg_dataset = file_to_dataset(trg_path)
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
    # iter = dataset.make_one_shot_iterator()
    iter = dataset.make_initializable_iterator()
    return (iter.initializer,) + iter.get_next()

class Model(object):
    def __init__(
            self,
            batch_size,
            trg_sos_id,
            trg_eos_id,
            src_vocab_size,
            trg_vocab_size,
            hidden_size,
            attention_hidden_size,
            layers,
            predict_max_step
    ):
        def network_define():
            initializer = tf.truncated_normal_initializer(stddev=0.1)
            with tf.variable_scope("input"):
                src_path = tf.placeholder(tf.string, shape=[])
                trg_path = tf.placeholder(tf.string, shape=[])
                iter_initializer, src_seq, src_size, trg_in_seq, trg_out_seq, trg_size\
                    = join_src_trg_dataset(src_path, trg_path, batch_size, trg_sos_id, trg_eos_id)
            with tf.variable_scope("embding", initializer=initializer):
                keep_prob = tf.placeholder(dtype=tf.float32, shape=[])
                enc_embeding = tf.get_variable("src_embeding", shape=[src_vocab_size, hidden_size])
                dec_embeding = tf.get_variable("trg_embeding", shape=[trg_vocab_size, hidden_size])
                src_embeded = tf.nn.dropout(tf.nn.embedding_lookup(enc_embeding, src_seq), keep_prob=keep_prob)
                trg_in_embeded = tf.nn.dropout(tf.nn.embedding_lookup(dec_embeding, trg_in_seq), keep_prob=keep_prob)

            with tf.variable_scope("cells", initializer=initializer):
                # enc_cell = tf.nn.rnn_cell.MultiRNNCell([
                #     tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for _ in range(layers)
                # ])
                enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
                enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
                dec_cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for _ in range(layers)
                ])

            with tf.variable_scope("encoder", initializer=initializer):
                # _, enc_state = tf.nn.dynamic_rnn(enc_cell, src_embeded, src_size, dtype=tf.float32)
                # enc_output: (2, batch_size, timestep, hidden_size)
                enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                    enc_cell_fw,
                    enc_cell_bw,
                    src_embeded,
                    src_size,
                    dtype=tf.float32
                )
                # enc_output: (batch, timestep, hidden_size*2)
                enc_output = tf.concat([enc_output[0],enc_output[1]], axis=-1)
            with tf.variable_scope("decoder", initializer=initializer):
                attention_mechanism = seq2seq.BahdanauAttention(
                    num_units=attention_hidden_size,
                    memory=enc_output,
                    memory_sequence_length=src_size
                )
                attention_cell = seq2seq.AttentionWrapper(
                    cell=dec_cell,
                    attention_mechanism=attention_mechanism,
                    attention_layer_size=attention_hidden_size
                )
                dec_output, _ = tf.nn.dynamic_rnn(attention_cell, trg_in_embeded, trg_size, dtype=tf.float32)
            with tf.variable_scope("softmax", initializer=initializer):
                # w_softmax = tf.transpose(dec_embeding)

                w_softmax = tf.get_variable("weight_softmax", shape=[attention_hidden_size,trg_vocab_size], dtype=tf.float32)
                b_softmax = tf.get_variable("bias_softmax", shape=[trg_vocab_size], dtype=tf.float32)
            with tf.variable_scope("loss", initializer=initializer):
                logits = tf.nn.bias_add(tf.matmul(tf.reshape(dec_output, shape=[-1, attention_hidden_size]), w_softmax), b_softmax)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.reshape(trg_out_seq, shape=[-1]),
                    logits=tf.reshape(logits, shape=[-1, trg_vocab_size])
                )
                mask = tf.reshape(tf.sequence_mask(trg_size, maxlen=tf.shape(trg_in_seq)[1], dtype=tf.float32), [-1])
                cost = tf.reduce_sum(loss * mask)
                token_cost = cost / tf.reduce_sum(mask)
            with tf.variable_scope("optimize", initializer=initializer):
                grads = tf.gradients(cost/tf.to_float(batch_size), tf.trainable_variables())
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0) # TODO
                train_op = tf.train.GradientDescentOptimizer(1.0).apply_gradients(zip(grads, tf.trainable_variables()))
            with tf.variable_scope("predict", initializer=initializer):
                def loop_cond(state, dec_ids, step):
                    return tf.reduce_all(tf.logical_and(
                        tf.not_equal(dec_ids.read(step), trg_eos_id),
                        tf.less(step, predict_max_step)
                    ))

                def loop_body(state, dec_ids, step):
                    prev_word_id = [dec_ids.read(step)]
                    prev_word_emb = tf.nn.embedding_lookup(dec_embeding, prev_word_id)
                    cell_output, cell_state = attention_cell.call(prev_word_emb, state)
                    output = tf.reshape(cell_output, [-1, attention_hidden_size])
                    logits = tf.reshape(tf.matmul(output, w_softmax) + b_softmax, shape=[-1])
                    dec_ids = dec_ids.write(step + 1, tf.argmax(logits, axis=0, output_type=tf.int32))
                    return cell_state, dec_ids, step + 1

                dec_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
                dec_ids = dec_ids.write(0, trg_sos_id)
                # loop_init = (attention_cell.zero_state(batch_size=1, dtype=tf.float32), dec_ids, 0) # 0 for step 0, and last effective index in trg_ids_tensor is 0
                debug_a = attention_cell.zero_state(batch_size=1, dtype=tf.float32)
                print(debug_a)
                _, debug_b = attention_cell.call(
                    tf.nn.embedding_lookup(dec_embeding, [trg_sos_id]),
                    debug_a
                )
            return debug_a,debug_b
        self.debug_a, self.debug_b = network_define()
        #         _, dec_ids, _ = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=loop_init)
        #     return src_path, trg_path, iter_initializer, src_seq, src_size, token_cost, train_op, dec_ids.stack(), keep_prob
        # self.src_path, self.trg_path, self.iter_initializer, self.src_seq, self.src_size,\
        #     self.token_cost, self.train_op, self.dec_ids, self.keep_prob = network_define()

    def train(self, model_path, src_path, trg_path):
        step = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(
                [tf.global_variables_initializer(), tf.local_variables_initializer()],
                feed_dict={self.src_path: src_path, self.trg_path: trg_path}
            )
            sess.run(
                self.iter_initializer,
                feed_dict={self.src_path: src_path, self.trg_path: trg_path}
            )
            for epoch in range(5):
                while True:
                    try:
                        co, _ = sess.run([self.token_cost, self.train_op], feed_dict={self.keep_prob: 0.8})
                        step += 1
                        print(step, co)
                        if step % 10 == 0:
                            saver.save(sess, model_path, global_step=step)
                    except tf.errors.OutOfRangeError: break
    def eval(self, model_path, en_vocab_file, zh_vocab_file):
        en_word_to_id = {}
        zh_id_to_word = {}
        with codecs.open(en_vocab_file, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                tokens = line.strip().split('\t')
                if len(tokens) != 2: continue
                en_word_to_id[tokens[0]] = int(tokens[1])
        with codecs.open(zh_vocab_file, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                tokens = line.strip().split('\t')
                if len(tokens) != 2: continue
                zh_id_to_word[int(tokens[1])] = tokens[0]
        unk_id = en_word_to_id['<unk>']
        input_sentense = "this is a test"
        input = [en_word_to_id.get(w, unk_id) for w in input_sentense.split(' ')]
        input_len = len(input)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(
                [tf.global_variables_initializer(), tf.local_variables_initializer()],
                feed_dict={self.src_path: "", self.trg_path: ""}
            )
            saver.restore(sess, model_path)
            zh = sess.run(
                self.dec_ids,
                feed_dict={self.src_seq: [input], self.src_size: [input_len], self.keep_prob: 1.0}
            )
            output_zh_sentense = ''.join([zh_id_to_word.get(w, "?") for w in zh])
            print(zh)
            print(output_zh_sentense)

if __name__ == "__main__":
    final_en_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.en.final"
    final_zh_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.zh.final"
    en_vocab_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.en.vocab"
    zh_vocab_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.zh.vocab"

    model = Model(
        batch_size=100,
        trg_sos_id=0,
        trg_eos_id=1,
        src_vocab_size=10000,
        trg_vocab_size=10000,
        hidden_size=1024,
        attention_hidden_size=100,
        layers=2,
        predict_max_step=100
    )
    model_path = globalconf.get_root() + "rnn/ted/seq2seq.model"
    with tf.Session() as sess:
        sess.run(tf.shape(model.debug_a))
        sess.run(tf.shape(model.debug_b))
    # model.train(model_path, final_en_file, final_zh_file)
    # model_path = globalconf.get_root() + "rnn/ted/seq2seq.model-140"
    # model.eval(model_path, en_vocab_file, zh_vocab_file)
