''' Extract OpenIE from input text with trained model. '''

#import sys
#sys.path.append('/home/zhengbaj/exp/supervised-oie/src')
#from trained_oie_extractor import Extraction
import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import numpy as np
from functools import reduce

from dataset import openie_collate_fn, OpenIEDataset
from transformer.Tagger import Tagger
from preprocess import convert_instance_to_idx_seq, concat_inp, SpacyParser

class Extraction:
    def __init__(self, sent, pred, args, probs,
                 calc_prob = lambda probs: 1.0 / (reduce(lambda x, y: x * y, probs) + 0.001)):
        self.sent = sent
        self.calc_prob = calc_prob
        self.probs = probs
        self.prob = self.calc_prob(self.probs)
        self.pred = pred
        self.args = args

    def __str__(self):
        return '\t'.join(
            map(str, [' '.join(self.sent), self.prob, '{}##{}'.format(*self.pred),
                      '\t'.join([' '.join(map(lambda x: x[0], arg)) +
                                 '##' + str(list(map(lambda x: x[1], arg))[0])
                                 for arg in self.args])]))

def tag2extraction(sent_list, pred_list, tag_prob_list):
    exts = []
    avg_conf = lambda probs: np.average(probs)
    for sent, pred, tag_probs in zip(sent_list, pred_list, tag_prob_list):
        tokens = sent.split(' ')
        pred_ind = pred.index(1)
        pred_word = tokens[pred_ind]
        cur_args, cur_arg, probs = [], [], []
        for i, (token, (tag, prob)) in enumerate(zip(tokens, tag_probs)):
            probs.append(prob)
            if tag.startswith('A'):
                cur_arg.append((token, i))
            elif cur_arg:
                cur_args.append(cur_arg)
                cur_arg = []
        if cur_arg:
            cur_args.append(cur_arg)
        # create extraction
        if cur_args:
            exts.append(Extraction(tokens, (pred_word, pred_ind), cur_args, probs, calc_prob=avg_conf))
    return exts

def read_instances_from_raw_sentence(inst_file, max_sent_len, keep_case):
    trimmed_sent_count = 0
    sent_count, cand_count = 0, 0
    raw_sent_insts, word_insts, pos_insts, \
    pred_word_insts, pred_pos_insts, pred_idx_insts = \
        [], [], [], [], [], []
    with open(inst_file) as f:
        for sent in f:
            sent_count += 1
            sent = sent.strip()
            word_inst = sent.split(' ')
            pos_inst = [t.tag_ for t in SpacyParser.parse_tokens(word_inst)][:max_sent_len]
            if not keep_case:
                word_inst = [word.lower() for word in word_inst]
            if len(word_inst) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = word_inst[:max_sent_len]
            for pred_ind, pos in zip(range(len(pos_inst)), pos_inst):
                if not pos.startswith('V'):
                    continue
                # one instance
                cand_count += 1
                pred_idx_inst = [0] * len(word_inst)
                pred_idx_inst[pred_ind] = 1
                raw_sent_insts.append(sent)
                word_insts.append(word_inst)
                pos_insts.append(pos_inst)
                pred_word_insts.append([word_inst[pred_ind] for _ in word_inst])
                pred_pos_insts.append([pos_inst[pred_ind] for _ in pos_inst])
                pred_idx_insts.append(pred_idx_inst)

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))
    print('[Info] #sentence {}, # candidates {}'.format(sent_count, cand_count))
    return raw_sent_insts, word_insts, pos_insts, pred_word_insts, pred_pos_insts, pred_idx_insts

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='openie_extract.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-sent', required=True,
                        help='Source sentence to extract from in raw format')
    parser.add_argument('-vocab', required=True,
                        help='training data which contains necessary information')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the extraction""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    test_raw_sent_insts, test_word_insts, test_pos_insts, \
    test_pred_word_insts, test_pred_pos_insts, test_pred_idx_insts = \
        read_instances_from_raw_sentence(
            opt.sent, preprocess_settings.max_word_seq_len,
            preprocess_settings.keep_case,)
    test_word_insts = convert_instance_to_idx_seq(
        test_word_insts, preprocess_data['word2idx'])
    test_pos_insts = convert_instance_to_idx_seq(
        test_pos_insts, preprocess_data['pos2idx'])
    test_pred_word_insts = convert_instance_to_idx_seq(
        test_pred_word_insts, preprocess_data['word2idx'])
    test_pred_pos_insts = convert_instance_to_idx_seq(
        test_pred_pos_insts, preprocess_data['pos2idx'])
    twc = concat_inp(test_word_insts, test_pos_insts, test_pred_idx_insts,
                     test_pred_word_insts, test_pred_pos_insts)

    test_loader = torch.utils.data.DataLoader(
        OpenIEDataset(
            word2idx=preprocess_data['word2idx'],
            tag2idx=preprocess_data['tag2idx'],
            word_insts=twc),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=openie_collate_fn)

    tagger = Tagger(opt)

    cur = 0
    with open(opt.output, 'w') as fout:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            probs, tags = tagger.tag_batch(*batch, skip_first=2) # skip PAD and UNK
            tag_probs = torch.cat([torch.unsqueeze(tags, -1).float(), torch.unsqueeze(probs, -1)], dim=-1)
            tag_probs = tag_probs.cpu().numpy()
            sent_list = test_raw_sent_insts[cur : cur + opt.batch_size]
            pred_list = test_pred_idx_insts[cur : cur + opt.batch_size]
            tag_prob_list = [[(test_loader.dataset.idx2tag[t], p) for t, p in tps] for tps in tag_probs]
            exts = tag2extraction(sent_list, pred_list, tag_prob_list)
            for ext in exts:
                fout.write('{}\n'.format(ext))
            cur += opt.batch_size
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
