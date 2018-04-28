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

11 - lstm max pooling, epochs 1-3 lr=1e-3, epoch 4 lr=1e-4
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 4

12 - lstm packed data, epochs 1-3 lr=1e-1, epocsh 4,5 1e-2, epoch 6 1e-3
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 6

13 - lstm packed data, epochs 1-3 lr=1e-1, epocsh 4,5 1e-2, epoch 6 1e-3, proportion_positive=.5,.25,.1
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 6

14 - lstm packed data, epochs 1-3 lr=1e-1, epocsh 4,5 1e-2, epoch 6 1e-3, proportion_positive=.2,.1,.05
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 6

15 - lstm packed data, epochs 1-3 lr=1e-1, epocsh 4,5 1e-2, epoch 6 1e-3, proportion_positive=.2,4/batch,2/batch
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 6

16 - lstm packed data, epochs 1-3 lr=1e-1, epocsh 4,5 1e-2, epoch 6 1e-3, proportion_positive=.2,.1,.05, lstm stack of 3, 512 units
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 6

17 - lstm packed data, epochs 1-3 lr=1e-1, epocsh 4,5 1e-2, epoch 6 1e-3, proportion_positive=.2,.1,.05, lstm stack of 3, 512 units - 2 final lstm layers
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 6

18 - lstm packed data, epochs 1-4 lr=1e-1, epocsh 5-7 1e-2, epochs 8-9 1e-3, epoch 10 1e-4, proportion_positive=.2,.1,.05, lstm stack of 3, 512 units
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 10

19 - lstm packed data, epochs 1-7 lr=1e-1, epocsh 8 1e-2, epochs 9 1e-3, epoch 10 1e-4, proportion_positive=.2,.1,.05, lstm stack of 3, 512 units
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 10

20 - lstm packed data, epochs 1-7 lr=1e-1, epocsh 8 1e-2, epochs 9 1e-3, epoch 10 1e-4, proportion_positive=.2,.1,.05, lstm stack of 3, 712 units
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 10

