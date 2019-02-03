''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from operator import itemgetter

def load_word_vector(filepath, is_binary=False, first_line=True):
    if is_binary:
        raise NotImplementedError
    words, vectors = [], []
    with open(filepath, 'r') as fin:
        first = True
        vocab_size, dim = None, None
        for i, l in enumerate(fin):
            if first and first_line:
                vocab_size, dim = map(int, l.split())
                first = False
                continue
            l = l.rstrip().split(' ')
            words.append(l[0])
            v = np.asarray(l[1:], dtype='float32')
            if dim is not None and len(v) != dim:
                raise Exception('word vector format error')
            vectors.append(v)
    words = np.array(words, dtype=str)
    vectors = np.array(vectors, dtype=np.float32)
    print('load word vector from {}'.format(filepath))
    return words, vectors

class WordVector(object):
    def __init__(self, filepath, is_binary=False, first_line=True, initializer='uniform'):
        if initializer not in {'uniform'}:
            raise Exception('initializer not supported')
        self.initializer = initializer
        self.raw_words, self.raw_vectors = \
            load_word_vector(filepath, is_binary=is_binary, first_line=first_line)
        self.raw_vocab_size = len(self.raw_words)
        self.raw_words2ind = dict(zip(self.raw_words, range(self.raw_vocab_size)))
        self.dim = self.raw_vectors.shape[1]
        self.vocab_size = self.raw_vectors.shape[0]
        self.words = np.array(self.raw_words)
        self.vectors = np.array(self.raw_vectors)

    def transform(self, new_words, oov_filepath=None):
        new_words = np.array(new_words)
        start_ind = self.raw_vocab_size
        def new_inder(w):
            nonlocal start_ind
            if w in self.raw_words2ind:
                return self.raw_words2ind[w]
            else:
                start_ind += 1
                return start_ind - 1
        new_ind = np.array([new_inder(w) for w in new_words])
        self.words = new_words
        print('total {} words, miss {} words'.format(
            len(new_words), start_ind - self.raw_vocab_size))
        if oov_filepath is not None:
            with open(oov_filepath, 'w') as fp:
                for i in range(len(new_words)):
                    if new_ind[i] >= self.raw_vocab_size:
                        fp.write('{}\n'.format(new_words[i]))
        if self.initializer == 'uniform':
            new_part = np.random.uniform(-.1, .1, [start_ind - self.raw_vocab_size, self.dim])
        else:
            raise ValueError
        self.vectors = np.concatenate([self.raw_vectors, new_part], axis=0)[new_ind]
        self.vocab_size = len(self.words)

    def get_vectors(self, normalize=False):
        if normalize:
            return self.vectors / np.sqrt(np.sum(self.vectors * self.vectors, axis=1, keepdims=True))
        else:
            return self.vectors

    def update(self, new_vectors):
        if new_vectors.shape != self.vectors.shape:
            raise Exception('shape is not correct')
        self.vectors = new_vectors

    def svd(self, n_components=10):
        svd = TruncatedSVD(n_components=n_components, algorithm='arpack')
        new_vectors = svd.fit_transform(self.vectors)
        self.dim = new_vectors.shape[1]
        self.vectors = new_vectors

    def save_to_file(self, filepath, is_binary=False):
        if is_binary:
            raise NotImplementedError
        with open(filepath, 'w') as fout:
            fout.write('{} {}\n'.format(self.vocab_size, self.dim))
            for i in range(self.vocab_size):
                fout.write('{} {}\n'.format(
                    self.words[i], ' '.join(map(lambda x: '{:.5f}'.format(x), self.vectors[i]))))

def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def read_instances_from_conll_csv(inst_file, max_sent_len, keep_case):
    ''' Convert conll file (in csv format) into word seq lists and tag seq lists '''
    df = pd.read_csv(inst_file, sep='\t', header=0, keep_default_na=False, quoting=3)

    # Split according to sentences
    sents = [df[df.run_id == run_id] for run_id in sorted(set(df.run_id.values))]

    word_insts, pred_insts, tag_insts = [], [], []
    trimmed_sent_count = 0
    useless_sent_count = 0
    for sent in sents:
        word_inst = [w if keep_case else w.lower() for w in sent.word.values]
        if len(word_inst) <= 0:
            continue
        if len(word_inst) > max_sent_len:
            trimmed_sent_count += 1
        word_inst = word_inst[:max_sent_len]
        pred = sent.head_pred_id.values[0]
        if pred >= len(word_inst):
            useless_sent_count += 1
            continue
        pred_inst = [0] * len(word_inst)
        pred_inst[pred] = 1
        tag_inst = sent.label.values[:max_sent_len].tolist()
        word_insts.append(word_inst)
        pred_insts.append(pred_inst)
        tag_insts.append(tag_inst)

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))
    if useless_sent_count > 0:
        print('[Warning] {} instances are useless under the max sentence length {}.'
              .format(useless_sent_count, max_sent_len))

    return word_insts, pred_insts, tag_insts


def build_vocab_idx(word_insts, min_word_count, word2idx=None):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    if word2idx is None:
        # default dict
        word2idx = {
            Constants.BOS_WORD: Constants.BOS,
            Constants.EOS_WORD: Constants.EOS,
            Constants.PAD_WORD: Constants.PAD,
            Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in sorted(word_count.items(), key=lambda x: -x[1]):
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main_mt(opt):
    ''' Main function for MT '''
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tgt_word_insts = read_instances_from_file(
        opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

def main_openie(opt):
    ''' Main function for OpenIE '''
    opt.max_token_seq_len = opt.max_word_seq_len # no need to include special tokens

    # Training set
    train_word_insts, train_pred_insts, train_tag_insts = \
        read_instances_from_conll_csv(opt.train_src, opt.max_word_seq_len, opt.keep_case)

    # Validation set
    valid_word_insts, valid_pred_insts, valid_tag_insts = \
        read_instances_from_conll_csv(opt.valid_src, opt.max_word_seq_len, opt.keep_case)

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data
        word2idx = predefined_data['dict']
        print('[Info] Pre-defined vocabulary found with {} tokens.'.format(len(word2idx)))
    else:
        print('[Info] Build vocabulary.')
        word2idx = build_vocab_idx(train_word_insts, opt.min_word_count)

    # Build tag classes
    # use PAD to mask out paddings or irrelevant subwords in BERT
    # use UNK to represent unseen classes
    print('[Info] Build classes.')
    tag2idx = build_vocab_idx(train_tag_insts, 0,
                              word2idx={
                                  Constants.PAD_WORD: Constants.PAD,
                                  Constants.UNK_WORD: Constants.UNK})
    print(['[Info] Classes:'])
    print(sorted(tag2idx.items(), key=lambda x: x[1]))
    opt.n_class = len(tag2idx)

    # word to index
    print('[Info] Convert word instances into sequences of word index.')
    train_word_insts = convert_instance_to_idx_seq(train_word_insts, word2idx)
    valid_word_insts = convert_instance_to_idx_seq(valid_word_insts, word2idx)

    # tag to index
    print('[Info] Convert tag instances into sequences of tag index.')
    train_tag_insts = convert_instance_to_idx_seq(train_tag_insts, tag2idx)
    valid_tag_insts = convert_instance_to_idx_seq(valid_tag_insts, tag2idx)

    # concatenate multiple word-related inputs
    concat_inp = lambda *x: [list(zip(*inst)) for inst in zip(*x)]
    data = {
        'settings': opt,
        'dict': word2idx,
        'train': {
            'word': concat_inp(train_word_insts, train_pred_insts),
            'tag': train_tag_insts},
        'valid': {
            'word': concat_inp(valid_word_insts, valid_pred_insts),
            'tag': valid_tag_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

def emb(opt):
    wv = WordVector(opt.pre_word_emb, is_binary=False, first_line=False, initializer='uniform')
    # save vocab of the raw embedding
    emb_vocab = wv.words.tolist()
    emb_vocab = [Constants.PAD_WORD, Constants.UNK_WORD] + emb_vocab # prepand two special tokens
    emb_vocab = dict(zip(emb_vocab, range(len(emb_vocab))))
    torch.save({'dict': emb_vocab}, opt.pre_word_emb + '.dict')
    if opt.vocab:
        print('use external vocab')
        vocab = torch.load(opt.vocab)['dict']
    else:
        print('use vocab of the raw embedding')
        vocab = emb_vocab
    words = list(map(itemgetter(0), sorted(vocab.items(), key=itemgetter(1))))
    print('high-frequency words: {}'.format(words[:10]))
    print('low-frequency words: {}'.format(words[-10:]))
    wv.transform(new_words=words)
    wv.save_to_file(opt.save_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src')
    parser.add_argument('-train_tgt')
    parser.add_argument('-valid_src')
    parser.add_argument('-valid_tgt')
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)
    parser.add_argument('-task', type=str, choices=['mt', 'openie', 'emb'], required=True)
    parser.add_argument('-pre_word_emb', type=str, default=None)

    opt = parser.parse_args()

    if opt.task == 'mt':
        main_mt(opt)
    elif opt.task == 'openie':
        main_openie(opt)
    elif opt.task == 'emb':
        emb(opt)
    else:
        raise ValueError
