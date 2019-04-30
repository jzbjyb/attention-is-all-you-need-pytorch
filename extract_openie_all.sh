save_model=$1
output_dir=$2
bs=64
vocab=data/oie2016/oie2016_path5_no_pred.low.pt

export CUDA_VISIBLE_DEVICES=3 &&
python openie_extract.py -model ${save_model}.chkpt -sent data/raw_sent/oie2016_test.txt \
    -vocab ${vocab} -output ${output_dir}/oie2016.txt -batch_size ${bs} &&
python openie_extract.py -model ${save_model}.chkpt -sent data/raw_sent/web.txt \
    -vocab ${vocab} -output ${output_dir}/web.txt -batch_size ${bs} &&
python openie_extract.py -model ${save_model}.chkpt -sent data/raw_sent/nyt.txt \
    -vocab ${vocab} -output ${output_dir}/nyt.txt -batch_size ${bs} &&
python openie_extract.py -model ${save_model}.chkpt -sent data/raw_sent/penn.txt \
    -vocab ${vocab} -output ${output_dir}/penn.txt -batch_size ${bs}
