import numpy as np
import tensorflow as tf
import globalconf
final_en_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.en.final"
final_zh_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.zh.final"

def file_to_dataset(file):
    dataset = tf.data.TextLineDataset(file)
    # input Shape must be rank 1 for 'StringSplit', output shape is rank 1
    return dataset.map(lambda line: tf.string_split([line]).values)\
            .map(lambda str_tokens: tf.string_to_number(string_tensor=str_tokens))

def join_src_trg_dataset(src_dataset, trg_dataset, batch_size, trg_sos_id, trg_eos_id):
    # rank 2: ([src list],[trg list])
    dataset = tf.data.Dataset.zip((src_dataset, trg_dataset))
    def extend_trg(src_list, trg_list):
        # rank 1
        trg_in = tf.concat([[trg_sos_id], trg_list], axis=0)
        trg_out = tf.concat([trg_list, [trg_eos_id]], axis=0)
        return (src_list, tf.size(src_list), trg_in, trg_out, tf.size(trg_out))
    return dataset.map(extend_trg).shuffle(10000).padded_batch(batch_size, padded_shapes=(
        (tf.TensorShape([None]), tf.TensorShape([]),
            tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]))
    ))

src_ds = file_to_dataset(final_en_file)
trg_ds = file_to_dataset(final_zh_file)
ds = join_src_trg_dataset(src_ds, trg_ds, 2, 0, 1)
iter = ds.make_one_shot_iterator()
line = iter.get_next()
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    for _ in range(1):
        print(sess.run(line))
