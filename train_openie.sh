export CUDA_VISIBLE_DEVICES=0 &&
python train.py -task openie -data data/oie2016/oie2016.low.pt \
	-save_model trained -save_mode best \
	-emb data/oie2016/w2v \
	-d_word_vec 50:50:50:0:0 -d_model 50 -d_inner_hid 50 -d_k 10 -d_v 10 -n_head 5 -n_layers 5 \
	-epoch 100 -batch_size 128 -n_warmup_steps 1000