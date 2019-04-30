#!/bin/sh
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=0

save_model=$1
output=$2

export CUDA_VISIBLE_DEVICES=3 &&
python openie_extract.py -model ${save_model}.chkpt -sent data/raw_sent/oie2016_test.txt \
    -vocab data/oie2016/oie2016_path5_no_pred.low.pt -output ${output} \
    -batch_size 256

