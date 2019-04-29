#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=0

python train.py -task openie -data data/oie2016/oie2016_path1_concat.low.pt \
	-save_model word_pred_300_test -save_mode best \
	-emb data/oie2016/w2v_300 \
	-d_word_vec 300:0:300:0:0 -d_model 300 -d_inner_hid 300 -d_k 60 -d_v 60 -n_head 5 -n_layers 4 \
	-epoch 100 -batch_size 256 -n_warmup_steps 1000 -rel_pos_emb_op no
