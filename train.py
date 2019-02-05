'''
This script handling the training process.
'''

import argparse
import math
import time
import contextlib

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import TranslationDataset, OpenIEDataset, paired_collate_fn, openie_paired_collate_fn
from transformer.Models import Transformer, TransformerTagger, count_parameters
from transformer.Optim import ScheduledOptim
from preprocess import WordVector

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    correct_mask = pred.eq(gold)
    n_correct = correct_mask.masked_select(non_pad_mask).sum().item()
    correct_pred = pred.masked_select(correct_mask & non_pad_mask)

    return loss, n_correct, correct_pred


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss

def tag_stat(tags):
    tags = tags.cpu().numpy()
    print(np.array(np.unique(tags, return_counts=True)).T)

def one_epoch(model, data, optimizer, device, smoothing, opt, is_train=False):
    if is_train:
        model.train()
        desc = '  - (Training)   '
    else:
        model.eval()
        desc = '  - (Validation)   '

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with contextlib.nullcontext() if is_train else torch.no_grad():
        for batch in tqdm(data, mininterval=2, desc=desc, leave=False):
            # prepare data
            if opt.task == 'mt':
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
                gold = tgt_seq[:, 1:]
            elif opt.task == 'openie':
                word_seq, pos_seq, tag_seq = map(lambda x: x.to(device), batch)
                gold = tag_seq
            else:
                raise ValueError

            # forward
            if is_train:
                optimizer.zero_grad()
            if opt.task == 'mt':
                pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            elif opt.task == 'openie':
                pred = model(word_seq, pos_seq)
                pred = pred.view(-1, pred.size(2)) # flatten
            else:
                raise ValueError

            # backward
            loss, n_correct, correct_pred = cal_performance(pred, gold, smoothing=smoothing)
            if is_train:
                loss.backward()

            # update parameters
            if is_train:
                optimizer.step_and_update_lr()

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy, correct_pred

def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_nll, train_accu, train_corr_pred = one_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing,
            opt=opt, is_train=True)
        train_ppl = math.exp(min(train_nll, 100))
        print('  - (Training) nll: {nll: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  nll=train_nll, ppl=train_ppl, accu=100*train_accu,
                  elapse=(time.time()-start)/60))
        if epoch_i % 10 == 0:
            tag_stat(train_corr_pred)

        start = time.time()
        valid_nll, valid_accu, valid_corr_pred = one_epoch(
            model, validation_data, optimizer, device, smoothing=opt.label_smoothing,
            opt=opt, is_train=False)
        valid_ppl = math.exp(min(valid_nll, 100))
        print('  - (Validation) nll: {nll: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    nll=valid_nll, ppl=valid_ppl, accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{nll: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, nll=train_nll, ppl=train_ppl, accu=100*train_accu))
                log_vf.write('{epoch},{nll: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, nll=valid_nll, ppl=valid_ppl, accu=100*valid_accu))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-emb', type=str, default=None)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    # used when we have multiple inputs (token, pos, pred in openie)
    parser.add_argument('-d_word_vec', type=str, default=None)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-task', type=str, choices=['mt', 'openie'], default='mt')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    if opt.task == 'mt':
        training_data, validation_data = prepare_dataloaders(data, opt)
        opt.src_vocab_size = training_data.dataset.src_vocab_size
        opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size
        if opt.embs_share_weight:
            assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
                'The src/tgt word2idx table are different but asked to share word embedding.'
    elif opt.task == 'openie':
        training_data, validation_data = prepare_dataloaders_openie(data, opt)
        opt.vocab_size = training_data.dataset.vocab_size
        opt.n_class = data['settings'].n_class
        opt.n_pos = data['settings'].n_pos
    else:
        raise ValueError

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    if opt.task == 'mt':
        opt.d_word_vec = opt.d_model
        transformer = Transformer(
            opt.src_vocab_size,
            opt.tgt_vocab_size,
            opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=opt.proj_share_weight,
            emb_src_tgt_weight_sharing=opt.embs_share_weight,
            d_k=opt.d_k,
            d_v=opt.d_v,
            d_model=opt.d_model,
            d_word_vec=opt.d_word_vec,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            dropout=opt.dropout).to(device)
    elif opt.task == 'openie':
        word_emb = None
        if opt.emb:
            word_emb = WordVector(opt.emb, is_binary=False, first_line=True, initializer='uniform').get_vectors()
            print('[Info] Use pretrained embedding with dim {}'.format(word_emb.shape[1]))
        # get dimensions
        # word, pos, pred_idx, pred_word, pred_pos
        if opt.d_word_vec:
            opt.d_vec_list = list(map(int, opt.d_word_vec.split(':')))
            assert opt.emb is None or (opt.d_vec_list[0] == word_emb.shape[1]), \
                'word vec dimension is not consistent'
        else:
            opt.d_word_vec = opt.d_model
            emb_dim = word_emb.shape[1] if word_emb is not None else opt.d_word_vec // 5
            pred_emb_dim = word_emb.shape[1] if word_emb is not None else opt.d_word_vec // 5
            rest_dim = opt.d_word_vec - emb_dim - pred_emb_dim
            pos_dim = rest_dim // 3
            pred_pos_dim = rest_dim // 3
            pred_idx_dim = rest_dim // 3
            opt.d_vec_list = [emb_dim, pos_dim, pred_idx_dim, pred_emb_dim, pred_pos_dim]
        print('[Info] input embedding dims: {}'.format(opt.d_vec_list))
        print('[Info] Transformer input dims: {}'.format(opt.d_model))
        for d in opt.d_vec_list:
            assert d >= 0, 'negative dimension'
        opt.n_cate_list = [opt.vocab_size, opt.n_pos, 2, opt.vocab_size, opt.n_pos]
        opt.emb_learnable_list = [d[0] and d[1] > 0 for d in
                                  zip([False, True, True, False, True], opt.d_vec_list)]
        opt.pre_emb_list = [d[0] if d[1] > 0 else None for d in
                            zip([word_emb, None, None, word_emb, None], opt.d_vec_list)]
        opt.emb_op = 'sum'
        transformer = TransformerTagger(
            opt.n_cate_list,
            opt.n_class,
            opt.max_token_seq_len,
            d_vec_list=opt.d_vec_list,
            pre_emb_list=opt.pre_emb_list,
            emb_op=opt.emb_op,
            emb_learnable_list=opt.emb_learnable_list,
            d_model=opt.d_model,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=opt.dropout).to(device)
    else:
        raise ValueError

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    print('[Info] #parameters: {}'.format(count_parameters(transformer)))
    train(transformer, training_data, validation_data, optimizer, device ,opt)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader

def prepare_dataloaders_openie(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        OpenIEDataset(
            word2idx=data['word2idx'],
            tag2idx=data['tag2idx'],
            word_insts=data['train']['word'],
            tag_insts=data['train']['tag']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=openie_paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        OpenIEDataset(
            word2idx=data['word2idx'],
            tag2idx=data['tag2idx'],
            word_insts=data['valid']['word'],
            tag_insts=data['valid']['tag']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=openie_paired_collate_fn)
    return train_loader, valid_loader

if __name__ == '__main__':
    main()
