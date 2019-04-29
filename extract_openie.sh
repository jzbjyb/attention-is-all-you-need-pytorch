#!/bin/sh
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=0

python openie_extract.py -model word_pred_300_test.chkpt -sent data/raw_sent/oie2016_test.txt \
    -vocab data/oie2016/oie2016_path1_concat.low.pt -output pred_test.txt \
    -batch_size 256
