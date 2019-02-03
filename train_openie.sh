python train.py -task openie -data data/oie2016/oie2016.low.pt \
	-save_model trained -save_mode best \
	-d_model 4 -d_inner_hid 4 -d_k 4 -d_v 4 -n_head 1 -n_layers=1 \
	-epoch 20 -batch_size 128