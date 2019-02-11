''' Handling the data io '''
import argparse
import re
import torch
import transformer.Constants as Constants
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from operator import itemgetter
import spacy
from spacy.tokens import Doc
import functools
from functools import lru_cache

def str2tokenlist(func, sep=' '):
    @functools.wraps(func)
    def wrapped(sent):
        return func(sent.split(sep))
    return wrapped

def tokenlist2str(func, sep=' '):
    @functools.wraps(func)
    def wrapped(tokens):
        return func(sep.join(tokens))
    return wrapped

class WhitespaceTokenizer(object):
    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        words = re.split(r' +', text) # Allow arbitrary number of spaces
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

class SpacyParser(object):
    #nlp = spacy.load('en_core_web_sm', create_make_doc=WhitespaceTokenizer) # python2.7
    nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def parse_sent(sent):
        ''' Parse a sentence (str) '''
        return SpacyParser.nlp(sent)

    @staticmethod
    @tokenlist2str
    @lru_cache(maxsize=10000)
    @str2tokenlist
    def parse_tokens(tokens):
        ''' Parse a token list '''
        tokens =  SpacyParser.nlp.tokenizer.tokens_from_list(tokens)
        return SpacyParser.nlp.tagger(tokens)

    @staticmethod
    @tokenlist2str
    @lru_cache(maxsize=10000)
    @str2tokenlist
    def get_pairwise_path(tokens):
        tokens = SpacyParser.nlp.tokenizer.tokens_from_list(tokens)
        n_t = len(tokens)
        path = [[[] for j in range(n_t)] for i in range(n_t)]
        for j, token in enumerate(SpacyParser.nlp.parser(tokens)):
            i = token.head.i
            if i == j:
                continue
            path[j][i].append(token.dep_)
            path[i][j].append('*' + token.dep_)
        return path

    @staticmethod
    def path_propagate(path, step=1):
        op = [[([c[0]] if len(c) == 1 else []) for c in r] for r in path]
        n_t = len(path)
        while step > 1:
            for i in range(n_t):
                for j in range(n_t):
                    if i == j:
                        continue
                    if len(path[i][j]) > 0:
                        continue
                    for k in range(n_t):
                        if len(path[i][k]) > 0 and len(op[k][j]) > 0:
                            path[i][j].extend(path[i][k])
                            path[i][j].append(op[k][j][0])
                            break
            step -= 1
        return path

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
        if initializer not in {'uniform', 'zero'}:
            raise Exception('initializer not supported')
        self.initializer = initializer
        self.first_line = first_line
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
        elif self.initializer == 'zero':
            new_part = np.zeros((start_ind - self.raw_vocab_size, self.dim))
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

    def save_to_file(self, filepath, is_binary=False, first_line=True):
        if is_binary:
            raise NotImplementedError
        with open(filepath, 'w') as fout:
            if first_line:
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

def gen_openie_sample(tokens, pred, max_sent_len, keep_case, get_pos=True,
                      pred_repeat=True, get_path=True, max_path_len=1):
    useless, trimmed = False, False
    word_inst, pred_inst, pred_word_inst, pred_pos_inst, pos_inst, path_inst = [None] * 6
    # POS
    if get_pos:
        pos_inst = [t.tag_ for t in SpacyParser.parse_tokens(tokens)]
        assert len(pos_inst) == len(tokens), \
            'part-of-speech tagging results should be of the same length as word sequence'
    # dependency parse
    if get_path:
        path_inst = SpacyParser.get_pairwise_path(tokens)
        SpacyParser.path_propagate(path_inst, step=len(tokens))
        assert len(path_inst) == len(tokens), \
            'dependency path results should be of the same length as word sequence'
    # tokens
    word_inst = [w if keep_case else w.lower() for w in tokens]
    if len(word_inst) <= 0:
        useless = True
        return useless, trimmed, word_inst, pred_inst, pred_word_inst, pred_pos_inst, pos_inst, path_inst
    if len(word_inst) > max_sent_len:
        trimmed = 1
    word_inst = word_inst[:max_sent_len]
    if get_pos:
        pos_inst = pos_inst[:max_sent_len]
    if get_path:
        path_inst = path_inst[:max_sent_len]
        path_inst = [[['->'.join(p[:max_path_len])] for p in pp[:max_sent_len]] for pp in path_inst]  # concat
        for i in range(len(path_inst)):
            path_inst[i][i] = ['<self>']
    # predicate
    if pred >= len(word_inst):
        useless = True
        return useless, trimmed, word_inst, pred_inst, pred_word_inst, pred_pos_inst, pos_inst, path_inst
    pred_inst = [1] * len(word_inst) # 0 is for PAD
    pred_inst[pred] = 2
    if pred_repeat:
        pred_word_inst = [word_inst[pred]] * len(word_inst)
        if get_pos:
            pred_pos_inst = [pos_inst[pred]] * len(pos_inst)
    return useless, trimmed, word_inst, pred_inst, pred_word_inst, pred_pos_inst, pos_inst, path_inst

def read_instances_from_conll_csv(inst_file, max_sent_len, keep_case,
                                  get_pos=True, pred_repeat=True, get_path=True, max_path_len=1):
    ''' Convert conll file (in csv format) into word seq lists and tag seq lists '''
    df = pd.read_csv(inst_file, sep='\t', header=0, keep_default_na=False, quoting=3)

    # Split according to sentences
    sents = [df[df.run_id == run_id] for run_id in sorted(set(df.run_id.values))]

    word_insts, pred_insts, pred_word_insts, pred_pos_insts, \
    pos_insts, path_insts, tag_insts = [], [], [], [], [], [], []
    trimmed_sent_count = 0
    useless_sent_count = 0
    for sent in sents:
        # retrieve from csv
        raw_tokens = sent.word.values.tolist()
        pred = sent.head_pred_id.values[0]
        tag_inst = sent.label.values[:max_sent_len].tolist()
        # get one sample
        s = gen_openie_sample(raw_tokens, pred, max_sent_len, keep_case, get_pos=get_pos,
                              pred_repeat=pred_repeat, get_path=get_path, max_path_len=max_path_len)
        if s[0]:
            useless_sent_count += 1
            continue
        if s[1]:
            trimmed_sent_count += 1
        # save
        word_inst, pred_inst, pred_word_inst, pred_pos_inst, pos_inst, path_inst = s[2:]
        word_insts.append(word_inst)
        pred_insts.append(pred_inst)
        pred_word_insts.append(pred_word_inst)
        pred_pos_insts.append(pred_pos_inst)
        pos_insts.append(pos_inst)
        path_insts.append(path_inst)
        tag_insts.append(tag_inst)

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))
    if useless_sent_count > 0:
        print('[Warning] {} instances are useless under the max sentence length {}.'
              .format(useless_sent_count, max_sent_len))
    return word_insts, pred_insts, pred_word_insts, pred_pos_insts,\
           pos_insts, path_insts, tag_insts

def build_path_vocab_idx(path_insts, min_word_count, word2idx=None):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in path_insts for pp in sent for p in pp for w in p)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    if word2idx is None:
        # default dict
        word2idx = {
            Constants.BOS_WORD: Constants.BOS,
            Constants.EOS_WORD: Constants.EOS,
            Constants.PAD_WORD: Constants.PAD,
            Constants.UNK_WORD: Constants.UNK}

    word2count = {w: 0 for w in full_vocab}

    for sent in path_insts:
        for pp in sent:
            for p in pp:
                for w in p:
                    word2count[w] += 1

    ignored_word_count = 0
    for word, count in sorted(word2count.items(), key=lambda x: -x[1]):
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx, word2count

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

    word2count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word2count[word] += 1

    ignored_word_count = 0
    for word, count in sorted(word2count.items(), key=lambda x: -x[1]):
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx, word2count

def convert_path_instance_to_idx_seq(path_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[[[word2idx.get(w, Constants.UNK) for w in p] for p in pp] for pp in s] for s in path_insts]

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def convert_pred_idx_to_repeat(idxs, lists):
    return [[list[idx]] * len(list) for idx, list in zip(idxs, lists)]

concat_inp = lambda *x: [list(zip(*inst)) for inst in zip(*x)]

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
            word2idx, _ = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx, _ = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx, _ = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

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
    train_word_insts, train_pred_idx_insts, train_pred_word_insts, train_pred_pos_insts,\
    train_pos_insts, train_path_insts, train_tag_insts = \
        read_instances_from_conll_csv(opt.train_src, opt.max_word_seq_len, opt.keep_case,
                                      get_pos=True, pred_repeat=True, get_path=True,
                                      max_path_len=opt.max_path_len)

    # Validation set
    valid_word_insts, valid_pred_idx_insts, valid_pred_word_insts, valid_pred_pos_insts,\
    valid_pos_insts, valid_path_insts, valid_tag_insts = \
        read_instances_from_conll_csv(opt.valid_src, opt.max_word_seq_len, opt.keep_case,
                                      get_pos=True, pred_repeat=True, get_path=True,
                                      max_path_len=opt.max_path_len)

    ## Build vocab
    # Build word vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data
        word2idx = predefined_data['dict']
        print('[Info] Pre-defined vocabulary found with {} tokens.'.format(len(word2idx)))
    else:
        print('[Info] Build vocabulary.')
        word2idx, _ = build_vocab_idx(train_word_insts, opt.min_word_count)

    # Build POS vocabulary
    print('[Info] POS vocabulary.')
    pos2idx, _ = build_vocab_idx(train_pos_insts, 0,
                              word2idx={
                                  Constants.PAD_WORD: Constants.PAD,
                                  Constants.UNK_WORD: Constants.UNK})
    print('[Info] POS:')
    print(sorted(pos2idx.items(), key=lambda x: x[1]))
    opt.n_pos = len(pos2idx)

    # Build path vocabulary
    print('[Info] PATH vocabulary.')
    path2idx, _ = build_path_vocab_idx(train_path_insts, 0, word2idx={
        Constants.PAD_WORD: Constants.PAD, Constants.UNK_WORD: Constants.UNK})
    print('[Info] PATH:')
    print(sorted(path2idx.items(), key=lambda x: x[1]))
    opt.n_path = len(path2idx)

    # Build tag classes
    # use PAD to mask out paddings or irrelevant subwords in BERT
    # use UNK to represent unseen classes
    print('[Info] Build classes.')
    tag2idx, tag2count = build_vocab_idx(train_tag_insts, 0,
                              word2idx={
                                  Constants.PAD_WORD: Constants.PAD,
                                  Constants.UNK_WORD: Constants.UNK})
    print('[Info] Classes and counts:')
    print(sorted(tag2idx.items(), key=lambda x: x[1]))
    tc = sorted(tag2count.items(), key=lambda x: -x[1])
    print(tc)
    print('[Info] most common tag {} takes {}'.format(
        tc[0][0], tc[0][1] / np.sum(list(map(itemgetter(1), tc)))))
    opt.n_class = len(tag2idx)

    ## Build pred vocabulary
    opt.n_pred_ind = 3 # 0/1 indicator with PAD

    ## Convert to index
    # word to index
    print('[Info] Convert word instances into sequences of word index.')
    train_word_insts = convert_instance_to_idx_seq(train_word_insts, word2idx)
    valid_word_insts = convert_instance_to_idx_seq(valid_word_insts, word2idx)
    train_pred_word_insts = convert_instance_to_idx_seq(train_pred_word_insts, word2idx)
    valid_pred_word_insts = convert_instance_to_idx_seq(valid_pred_word_insts, word2idx)

    # pos to index
    print('[Info] Convert pos instances into sequences of pos index.')
    train_pos_insts = convert_instance_to_idx_seq(train_pos_insts, pos2idx)
    valid_pos_insts = convert_instance_to_idx_seq(valid_pos_insts, pos2idx)
    train_pred_pos_insts = convert_instance_to_idx_seq(train_pred_pos_insts, pos2idx)
    valid_pred_pos_insts = convert_instance_to_idx_seq(valid_pred_pos_insts, pos2idx)

    # path to index
    print('[Info] Convert path instances into sequences of path index.')
    train_path_insts = convert_path_instance_to_idx_seq(train_path_insts, path2idx)
    valid_path_insts = convert_path_instance_to_idx_seq(valid_path_insts, path2idx)

    # tag to index
    print('[Info] Convert tag instances into sequences of tag index.')
    train_tag_insts = convert_instance_to_idx_seq(train_tag_insts, tag2idx)
    valid_tag_insts = convert_instance_to_idx_seq(valid_tag_insts, tag2idx)


    # concatenate multiple word-related inputs
    twc = concat_inp(train_word_insts, train_pos_insts, train_pred_idx_insts,
                     train_pred_word_insts, train_pred_pos_insts)
    vwc = concat_inp(valid_word_insts, valid_pos_insts, valid_pred_idx_insts,
                     valid_pred_word_insts, valid_pred_pos_insts)
    data = {
        'settings': opt,
        'word2idx': word2idx,
        'pos2idx': pos2idx,
        'tag2idx': tag2idx,
        'path2idx': path2idx,
        'train': {
            'word': twc,
            'path': train_path_insts,
            'tag': train_tag_insts},
        'valid': {
            'word': vwc,
            'path': valid_path_insts,
            'tag': valid_tag_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

def emb(opt, first_line=False):
    wv = WordVector(opt.pre_word_emb, is_binary=False, first_line=first_line, initializer='zero')
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
    wv.save_to_file(opt.save_data, first_line=True)

def emb_svd(opt, first_line=False, dim=10):
    wv = WordVector(opt.pre_word_emb, is_binary=False, first_line=first_line, initializer='uniform')
    wv.svd(n_components=dim)
    wv.save_to_file(opt.save_data, first_line=first_line)

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
    parser.add_argument('-task', type=str, choices=['mt', 'openie', 'emb', 'emb_svd'], required=True)
    parser.add_argument('-pre_word_emb', type=str, default=None)

    opt = parser.parse_args()

    if opt.task == 'mt':
        main_mt(opt)
    elif opt.task == 'openie':
        opt.get_pos = True
        opt.get_path = True
        opt.pred_repeat = True
        opt.max_path_len = 1
        main_openie(opt)
    elif opt.task == 'emb':
        # python preprocess.py -task emb -pre_word_emb X [-vocab X] -save_data X
        emb(opt, first_line=False)
    elif opt.task == 'emb_svd':
        # python preprocess.py -task emb_svd -pre_word_emb X -save_data X
        emb_svd(opt, first_line=False, dim=16)
    else:
        raise ValueError
