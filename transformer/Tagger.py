''' This module will handle the tagging. '''

import torch

from transformer.Models import TransformerTagger

class Tagger(object):
    ''' Load with trained model and handle tagging '''

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

        model = TransformerTagger(
            model_opt.n_cate_list,
            model_opt.n_class,
            model_opt.max_token_seq_len,
            d_vec_list=model_opt.d_vec_list,
            pre_emb_list=model_opt.pre_emb_list,
            emb_op=model_opt.emb_op,
            emb_learnable_list=model_opt.emb_learnable_list,
            d_model=model_opt.d_model,
            d_inner=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            dropout=model_opt.dropout)

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        model = model.to(self.device)

        self.model = model
        self.model.eval()

    def tag_batch(self, word_seq, pos_seq, skip_first=2):
        ''' tagging work in one batch '''
        with torch.no_grad():
            word_seq, pos_seq = word_seq.to(self.device), pos_seq.to(self.device)
            pred = self.model(word_seq, pos_seq)
            # the first skip_first classes are useless (PAD or UNK) in prediction
            pred = torch.nn.Softmax(-1)(pred[:, :, skip_first:]) # logit to prob
            pred = pred.max(-1)
            return pred[0], pred[1] + skip_first
