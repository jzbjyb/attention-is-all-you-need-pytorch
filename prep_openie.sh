python preprocess.py -task openie -train_src data/oie2016/train.conll \
	-valid_src data/oie2016/dev.conll -save_data data/oie2016/oie2016.low.pt \
	-vocab data/oie2016/glove.6B.50d.txt.dict -max_len 64 -min_word_count 0