save_model=$1

export CUDA_VISIBLE_DEVICES=3 &&
python train.py -task openie -data data/oie2016/oie2016_path5_no_pred.low.pt \
	-save_model ${save_model} -save_mode best \
	-emb data/oie2016/w2v_50 \
	-d_word_vec 50:10:10:0:0 -d_model 70 -d_inner_hid 70 -d_k 14 -d_v 14 -n_head 5 -n_layers 4 \
	-emb_op concat -rel_pos_emb_op no -dropout 0.1 \
	-epoch 100 -batch_size 256 -n_warmup_steps 200