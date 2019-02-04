python train.py -task openie -data data/oie2016/oie2016.low.pt \
	-save_model trained -save_mode best \
	-d_word_vec 50:10:50 -d_model 110 -d_inner_hid 110 -d_k 110 -d_v 110 -n_head 1 -n_layers=1 \
	-epoch 20 -batch_size 128