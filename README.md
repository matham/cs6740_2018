# cs6740_2018


1 - hinge loss
python cs6740/train_img.py --dataRoot data --save result --batchSz 512 --textEmbeddingSize 300

2- cos loss
python cs6740/train_img.py --dataRoot data --save result --batchSz 512 --textEmbeddingSize 300

3 - cos loss - crash after first epoch
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 100

4 - cos loss - val subset
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 100 --valSubset cs6740/data/coco_val_subset.txt
