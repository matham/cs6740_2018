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

21 - lstm packed data, epochs 1-21 lr=1e-1, epocsh 22-24 1e-2, epochs 25-27 1e-3, epoch 28-29 1e-4, proportion_positive=.2,.1,.05, lstm stack of 3, 512 units
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 29

22 - lstm packed data, epochs 1-11 lr=1e-1, epocsh 12-27 1e-2, epochs 28-37 1e-3, epoch 38-43 1e-4, proportion_positive=.2,.1,.05, lstm stack of 3, 512 units
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 43

23 - lstm packed data, epochs 1-6 lr=1e-1, epocsh 7-11 1e-2, epochs 12-14 1e-3, epoch 15-16 1e-4, proportion_positive=.2,.1,.05, lstm stack of 3, 512 units, initial layer before lstm
python cs6740/train_img.py --dataRoot data --save result --batchSz 256 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 16

24 - lstm packed data, epochs 1-21 lr=1e-1, epocsh 22-24 1e-2, epochs 25-27 1e-3, epoch 28-29 1e-4, proportion_positive=.2,.1,.05, lstm stack of 3, 512 units
python cs6740/train_img.py --dataRoot data --save result --batchSz 32 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 29

25 - lstm packed data, epochs 1-11 lr=1e-1, epocsh 12-27 1e-2, epochs 28-37 1e-3, epoch 38-43 1e-4, proportion_positive=.2,.1@3, lstm stack of 3, 512 units, initial layer before lstm
python cs6740/train_img.py --dataRoot data --save result --batchSz 64 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 43

26 - lstm packed data, epochs 1-11 lr=1e-1, epocsh 12-27 1e-2, epochs 28-37 1e-3, epoch 38-43 1e-4, proportion_positive=.2,.1@3, lstm stack of 3, 712 units, initial layer before lstm
python cs6740/train_img.py --dataRoot data --save result --batchSz 64 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 43

27 - lstm packed data, epochs 1-11 lr=1e-1, epocsh 12-27 1e-2, epochs 28-37 1e-3, epoch 38-43 1e-4, proportion_positive=.2,.1@3, lstm stack of 3, 512 units
python cs6740/train_img.py --dataRoot data --save result --batchSz 64 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel lstm --nEpochs 43

28 - lstm packed data, epochs 1-8 lr=1e-1, epocsh 9-16 1e-2, epochs 17-20 1e-3, epoch 21-24 1e-4, proportion_positive=.2,.1@3, bow
python cs6740/train_img.py --dataRoot data --save result --batchSz 64 --textEmbeddingSize 300 --valSubset cs6740/data/coco_val_subset.txt --textModel bow --nEpochs 24

29 - genome dataset fine-tuned on 22
python cs6740/train_img.py --dataRoot data --save result --batchSz 64 --textEmbeddingSize 300 --textModel lstm --nEpochs 10 --preTrainedModel logs/27/model_last_epoch.t7 --genomeLong

