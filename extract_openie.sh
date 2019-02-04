python openie_extract.py -model trained.chkpt -sent data/raw_sent/oie2016_test.txt \
    -vocab data/oie2016/oie2016.low.pt -output pred_test.txt \
    -batch_size 64