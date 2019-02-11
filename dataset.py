import numpy as np
import torch
import torch.utils.data

from transformer import Constants

def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)

def openie_paired_collate_fn(insts):
    word_insts, path_insts, tag_insts = list(zip(*insts))
    return (*collate_fn(word_insts), collate_fn_2d(path_insts),
            collate_fn(tag_insts, get_pos=False))

def openie_collate_fn(insts):
    word_insts, path_insts = list(zip(*insts))
    return (*collate_fn(word_insts), collate_fn_2d(path_insts))

def collate_fn_2d(insts):
    if len(insts) <= 0:
        raise ValueError
    max_len = max(len(inst) for inst in insts)
    max_depth = max(len(c) for inst in insts for r in inst for c in r)
    insts = [[[c + [Constants.PAD] * (max_depth - len(c)) for c in r] for r in inst] for inst in insts]
    pad_shape = (max_len, max_len, max_depth)
    def pad(inst):
        ninst = np.full(pad_shape, Constants.PAD)
        ninst[:len(inst), :len(inst)] = inst
        return ninst
    batch_seq = np.array([pad(inst) for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)
    return batch_seq

def collate_fn(insts, get_pos=True):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(inst) for inst in insts)
    # get the size of one element
    # a element could be either a list of integers or an integer
    ele_size = None
    for inst in insts:
        if len(inst) == 0:
            continue
        ele_size = inst[0]
        if hasattr(ele_size, '__len__'):
            ele_size = len(ele_size)
            if ele_size == 0:
                raise ValueError('empty element')
        else:
            ele_size = 0
        break
    if ele_size is None:
        raise ValueError('all instances are empty')

    batch_seq = np.array([
        inst + [[Constants.PAD] * ele_size or Constants.PAD] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    if get_pos:
        batch_pos = np.array([
            [pos_i+1 if (w_i[0] if ele_size else w_i) != Constants.PAD else 0
             for pos_i, w_i in enumerate(inst)] for inst in batch_seq]) # the first element is word
        batch_pos = torch.LongTensor(batch_pos)
        return batch_seq, batch_pos

    return batch_seq

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self, src_word2idx, tgt_word2idx,
        src_insts=None, tgt_insts=None):

        assert src_insts
        assert not tgt_insts or (len(src_insts) == len(tgt_insts))

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._src_insts = src_insts

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_insts:
            return self._src_insts[idx], self._tgt_insts[idx]
        return self._src_insts[idx]

class OpenIEDataset(torch.utils.data.Dataset):
    def __init__(
        self, word2idx, tag2idx, word_insts=None, path_insts=None, tag_insts=None):

        assert word_insts
        assert path_insts

        idx2word = {idx:word for word, idx in word2idx.items()}
        idx2tag = {idx:tag for tag, idx in tag2idx.items()}
        self._word2idx = word2idx
        self._idx2word = idx2word
        self._tag2idx = tag2idx
        self._idx2tag = idx2tag
        self._word_insts = word_insts
        self._tag_insts = tag_insts
        self._path_insts = path_insts

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._word_insts)

    @property
    def vocab_size(self):
        ''' Property for vocab size '''
        return len(self._word2idx)

    @property
    def word2idx(self):
        ''' Property for word dictionary '''
        return self._word2idx

    @property
    def idx2word(self):
        ''' Property for index dictionary '''
        return self._idx2word

    @property
    def tag2idx(self):
        ''' Property for word dictionary '''
        return self._tag2idx

    @property
    def idx2tag(self):
        ''' Property for index dictionary '''
        return self._idx2tag

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        # return word seq, path seq, tag seq separately
        if self._tag_insts:
            return self._word_insts[idx], self._path_insts[idx], self._tag_insts[idx]
        return self._word_insts[idx], self._path_insts[idx]
