import globalconf
import codecs
from nltk.tokenize import word_tokenize
import jieba

# https://wit3.fbk.eu/mt.php?release=2015-01
in_en_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.en"
in_zh_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.zh"

token_en_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.en.token"
token_zh_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.zh.token"

vocab_en_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.en.vocab"
vocab_zh_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.zh.vocab"

id_en_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.en.id"
id_zh_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.zh.id"

final_en_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.en.final"
final_zh_file = globalconf.get_root() + "rnn/ted/en-zh/train.tags.en-zh.zh.final"

def tokenize():
    stop_word = [' ']
    with codecs.open(in_en_file, mode='r', encoding='utf8') as f_in,\
            codecs.open(token_en_file, mode='w', encoding='utf8') as f_out:
        lines = f_in.readlines()
        for line in lines:
            f_out.write(' '.join(word_tokenize(line.strip().lower())) + "\n")

    with codecs.open(in_zh_file, mode='r', encoding='utf8') as f_in,\
            codecs.open(token_zh_file, mode='w', encoding='utf8') as f_out:
        lines = f_in.readlines()
        for line in lines:
            tokens = jieba.lcut(line.strip(), cut_all=False)
            tokens = [w for w in tokens if w not in stop_word]
            f_out.write(' '.join(tokens) + "\n")


def build_vocab(token_file, save_path, keep_size):
    word_count = {}
    with codecs.open(token_file, mode='r', encoding='utf8') as f:
        totalDoc = ' '.join([l.strip() for l in f.readlines()])
        for w in totalDoc.split(' '):
            word_count[w] = word_count.get(w, 0) + 1
    print(len(word_count))
    sorted_word = sorted(word_count.items(), key=(lambda e: e[1]), reverse=True)
    sorted_word = ['<sos>','<eos>','<unk>'] + [e[0] for e in sorted_word]
    vocab = {}
    with codecs.open(save_path, mode='w', encoding='utf8') as f:
        wid = 0
        for w in sorted_word[:keep_size]:
            f.write(w + "\t" + str(wid) + "\n")
            vocab[w] = str(wid)
            wid += 1
    return vocab


def tokens_to_ids(token_file, id_file, vocab):
    unk_id = vocab['<unk>']
    with codecs.open(token_file, mode='r', encoding='utf8') as f_in,\
            codecs.open(id_file, mode='w', encoding='utf8') as f_out:
        for raw in f_in.readlines():
            id_list = [vocab.get(w, unk_id) for w in raw.strip().split(' ')]
            f_out.write(' '.join(id_list) + "\n")

def filter_by_length(id_en_file, id_zh_file, final_en_file, final_zh_file, min_length, max_length):
    with codecs.open(id_en_file, mode='r', encoding='utf8') as f:
        id_en_lines = [line.strip() for line in f.readlines()]
    with codecs.open(id_zh_file, mode='r', encoding='utf8') as f:
        id_zh_lines = [line.strip() for line in f.readlines()]

    with codecs.open(final_en_file, mode='w', encoding='utf8') as f_en,\
            codecs.open(final_zh_file, mode='w', encoding='utf8') as f_zh:
        for line_en, line_zh in zip(id_en_lines, id_zh_lines):
            len_en = len(line_en.split(' '))
            len_zh = len(line_zh.split(' '))
            if len_en < min_length or len_en > max_length: continue
            if len_zh < min_length or len_zh > max_length: continue
            f_en.write(line_en + "\n")
            f_zh.write(line_zh + "\n")

# tokenize()
vocab = build_vocab(token_en_file, vocab_en_file, 10000)
tokens_to_ids(token_en_file, id_en_file, vocab)

vocab = build_vocab(token_zh_file, vocab_zh_file, 10000)
tokens_to_ids(token_zh_file, id_zh_file, vocab)

filter_by_length(id_en_file, id_zh_file, final_en_file, final_zh_file, 2, 50)
