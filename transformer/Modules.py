import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"


class LSTMRelPosEmb(nn.Module):
    ''' Embed a relative position sequence (path) using LSTM '''

    def __init__(self, emb, dropout=0.1, padding_idx=0):
        super().__init__()
        self.pad_idx = padding_idx
        self.emb = emb
        self.emb_d = emb.embedding_dim
        self.h_dim = emb.embedding_dim
        self.n_layer = 1
        self.lstm = nn.LSTM(self.emb_d, self.h_dim, num_layers=self.n_layer,
                            dropout=dropout, bidirectional=False)

    def forward(self, rel_pos_seq):
        rpsl = torch.sum(rel_pos_seq.ne(self.pad_idx).type(torch.int), -1)
        rel_pos_seq = self.emb(rel_pos_seq)
        b, l, l, sl, es = rel_pos_seq.size()
        bs = b * l * l
        rel_pos_seq = rel_pos_seq.view(bs, sl, es)
        rpsl = rpsl.view(bs)

        # sort
        rpsl, idx = rpsl.sort(0, descending=True)
        _, ridx = idx.sort(0, descending=False)
        rel_pos_seq = rel_pos_seq[idx]
        # transpose
        rel_pos_seq = torch.transpose(rel_pos_seq, 0, 1)  # seq_len first
        rpsl += torch.eq(rpsl, 0).type(rpsl.dtype) # pretend to have at least one word

        bucket = 512
        ht_li = []
        for i in range(0, bs, bucket):
            trps = rel_pos_seq[:, i:i+bucket]
            tbs = trps.size(1)
            trpsl = rpsl[i:i+bucket]
            trps = nn.utils.rnn.pack_padded_sequence(trps, trpsl, batch_first=False)
            h0 = torch.zeros((self.n_layer, tbs, self.h_dim)).to('cuda')
            c0 = torch.zeros((self.n_layer, tbs, self.h_dim)).to('cuda')
            _, (ht, ct) = self.lstm(trps, (h0, c0)) # ht is n_layer x bs x h_dim
            ht_li.append(ht)
        ht = torch.cat(ht_li, dim=1)
        rel_pos = torch.transpose(ht, 0, 1)
        rel_pos = rel_pos[ridx].view(b, l, l, -1)
        return rel_pos

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class RelScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention with relative positional encoding '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, rel_pos, rel_pos_v=None, mask=None):
        nb, lq, dk = q.size()
        nb, lk, dk = k.size()
        nb, lv, dv = v.size()
        nb, lq, lk, dk = rel_pos.size()

        # TODO: add param
        # only use rel_pos
        #attn = torch.einsum('nqd,nqkd->nqk', q, rel_pos)  # (n*b) x lq x lk
        # combine word and rel_pos
        #k = k.view(nb, 1, lk, dk)
        #k = k + rel_pos
        #attn = torch.einsum('nqd,nqkd->nqk', q, k) # (n*b) x lq x lk
        # only use word
        attn = torch.bmm(q, k.transpose(1, 2))

        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        
        # TODO: add param
        if rel_pos_v is not None:
            # only use rel_pos
            output = torch.einsum('nqk,nqkd->nqd', attn, rel_pos_v)
            # combine word and rel_pos
            #v = rel_pos_v + v.view(nb, 1, lv, dv)
            #output = torch.einsum('nqk,nqkd->nqd', attn, v)
        else:
            # only use word
            output = torch.bmm(attn, v)

        return output, attn
