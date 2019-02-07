''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD) # b x lk
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class MultiInpEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_cate_list, len_max_seq, d_vec_list, pre_emb_list, emb_learnable_list,
            emb_op, rel_pos_emb_op, n_rel_pos, n_layers, n_head, d_k, d_v, d_model, d_inner,
            dropout=0.1):

        super().__init__()

        assert len(pre_emb_list) == len(n_cate_list) \
            and len(pre_emb_list) == len(d_vec_list) \
            and len(pre_emb_list) == len(emb_learnable_list), \
            'Number of elements should be the same.'

        assert emb_op in {'sum', 'concat'}, 'emb_op not supported'
        assert rel_pos_emb_op in {'no', 'lookup', 'lstm', 'trans'}, \
            'rel_pos_emb_op not supported'

        n_position = len_max_seq + 1

        # non empty input indicator
        self.inp_use_list = [d > 0 for d in d_vec_list]
        # number of non empty input
        self.n_inp_use = np.sum(self.inp_use_list)
        # dimension check (d_inp is the dimension after embedding is applied)
        self.emb_op = emb_op
        if self.emb_op == 'sum':
            d_inp = 0
            for d in d_vec_list:
                if d and not d_inp:
                    d_inp = d
                if d and d_inp and d != d_inp:
                    raise ValueError('input dimensions are consistent')
            if not d_inp:
                raise ValueError('input dimensions are all zeros')
        elif self.emb_op == 'concat':
            d_inp = np.sum(d_vec_list) # all the input embeddings are concatenated
        else:
            raise ValueError
        assert d_model == d_inp, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        # pretrained embeding check
        for n_cate, d_vec, emb, use in zip(n_cate_list, d_vec_list, pre_emb_list, self.inp_use_list):
            if use and emb is not None and emb.shape != (n_cate, d_vec):
                raise ValueError('pretrained embedding dimension is wrong')

        # build embedding
        self.word_emb_list = nn.ModuleList([
            nn.Embedding(n_cate, d_vec, padding_idx=Constants.PAD) if emb is None \
                else nn.Embedding.from_pretrained(torch.FloatTensor(emb))
            for n_cate, d_vec, emb, use in
            zip(n_cate_list, d_vec_list, pre_emb_list, self.inp_use_list) if use])

        # set embedding grad
        emb_learnable_list = [el for el, use in zip(emb_learnable_list, self.inp_use_list)]
        for i, emb in enumerate(self.word_emb_list):
            emb.weight.requires_grad = emb_learnable_list[i]

        # build abs positional embedding
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_inp, padding_idx=0),
            freeze=True)

        # build rel positional embedding
        self.rel_pos_emb_op = rel_pos_emb_op
        if self.rel_pos_emb_op == 'no':
            rel_pos_op = None
        elif self.rel_pos_emb_op == 'lookup':
            rel_pos_op = 'external'
            self.rel_pos_emb = nn.Embedding(n_rel_pos, d_inp, padding_idx=Constants.PAD)
        else:
            raise NotImplementedError

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v,
                         rel_pos_op=rel_pos_op, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, word_seqs, pos_seq, rel_pos_seq=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Split inputs
        word_seqs = [torch.squeeze(c, dim=-1) for c, use in
            zip(torch.chunk(word_seqs, word_seqs.size(-1), dim=-1), self.inp_use_list) if use]

        # -- Prepare masks
        # the first inp is always tokens
        slf_attn_mask = get_attn_key_pad_mask(seq_k=word_seqs[0], seq_q=word_seqs[0])
        non_pad_mask = get_non_pad_mask(word_seqs[0])

        # -- Embedding
        enc_output = [emb(word_seq) for word_seq, emb in zip(word_seqs, self.word_emb_list)]
        if self.emb_op == 'sum':
            enc_output = torch.sum(torch.stack(enc_output, dim=-1), dim=-1)
        elif self.emb_op == 'concat':
            enc_output = torch.cat(enc_output, dim=-1)
        else:
            raise ValueError
        enc_output += self.position_enc(pos_seq)

        # -- Rel pos embedding
        if self.rel_pos_emb_op == 'no':
            rel_pos_seq = None
        elif self.rel_pos_emb_op == 'lookup':
            rel_pos_seq = self.rel_pos_emb(rel_pos_seq)
            rel_pos_seq = rel_pos_seq.mean(-2)
        else:
            raise NotImplementedError

        # -- Forward
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, rel_pos_seq,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))

class TransformerTagger(nn.Module):
    ''' A sequence tagging model. '''

    def __init__(
            self,
            n_cate_list, n_class, len_max_seq, d_vec_list,
            pre_emb_list, emb_learnable_list, emb_op,
            rel_pos_emb_op, n_rel_pos,
            d_model=512, d_inner=2048, n_layers=6,
            n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()

        self.encoder = MultiInpEncoder(
            n_cate_list=n_cate_list, len_max_seq=len_max_seq,
            d_vec_list=d_vec_list, pre_emb_list=pre_emb_list,
            emb_learnable_list=emb_learnable_list, emb_op=emb_op,
            rel_pos_emb_op=rel_pos_emb_op, n_rel_pos=n_rel_pos,
            d_model=d_model,d_inner=d_inner, n_layers=n_layers,
            n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.word_prj = nn.Linear(d_model, n_class, bias=False)
        nn.init.xavier_normal_(self.word_prj.weight)

    def forward(self, inps, pos, rel_pos=None):

        enc_output, *_ = self.encoder(inps, pos, rel_pos)
        tag_logit = self.word_prj(enc_output)

        return tag_logit