import globalconf
import codecs

def genVocab(file):
    with codecs.open(file) as f:
        word_count = {}
        while True:
            line = f.readline()
            if not line: break
            words = line.strip().split(' ')
            for w in words:
                word_count[w] = word_count.get(w, 0) + 1
        sorted_word_by_cnt = sorted(word_count.items(), key=(lambda tuple: tuple[1]), reverse=True)
        sorted_word_by_cnt = ['<eos>'] + [x[0] for x in sorted_word_by_cnt]
        id = 0
        word_id_dict = {}
        for w in sorted_word_by_cnt:
            word_id_dict[w] = str(id)
            # print(id, w)
            id += 1
    return word_id_dict


def word_to_id(in_file, out_file, vocab):
    unk_id = vocab['<unk>']
    with codecs.open(in_file, mode='r', encoding='utf8') as f_in, codecs.open(out_file, mode='w', encoding='utf8') as f_out:
        while True:
            line = f_in.readline()
            if not line: break
            words = line.strip().split(' ') + ['<eos>']
            ids = [vocab.get(w, unk_id) for w in words]
            f_out.write(' '.join(ids) + "\n")


in_train_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/ptb.train.txt"
in_valid_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/ptb.valid.txt"
in_test_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/ptb.test.txt"

out_train_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/id.train.txt"
out_valid_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/id.valid.txt"
out_test_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/id.test.txt"
out_vacab_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/vocab.txt"

vocab = genVocab(in_train_file)
with codecs.open(out_vacab_file, mode='w', encoding='utf8') as f_vocab:
    for w, id in vocab.items():
        f_vocab.write(str(id) + "\t" + w + "\n")
word_to_id(in_train_file, out_train_file, vocab)
word_to_id(in_valid_file, out_valid_file, vocab)
word_to_id(in_test_file, out_test_file, vocab)
