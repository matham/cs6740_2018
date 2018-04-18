# cs6740_2018


1 - hinge loss
python cs6740/train_img.py --dataRoot data --save result --batchSz 512 --textEmbeddingSize 300

2- cos loss
python cs6740/train_img.py --dataRoot data --save result --batchSz 512 --textEmbeddingSize 300

3 - cos loss - crash after first epoch
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 100

4 - cos loss - val subset
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 100 --valSubset cs6740/data/coco_val_subset.txt

5 - lstm added
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm

6 - lstm one direction, 100 units in hidden layer
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 7

7 - lstm one direction, dropout=.1, 4 stacks, no normalizing
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 7

8 - bow
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel bow --nEpochs 7

9 - lstm with max pooling
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 7

10 - lstm last output, epochs 1-3 lr=1e-3, epoch 4 lr=1e-4
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 4

