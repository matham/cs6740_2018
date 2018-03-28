#!/usr/bin/env python3

from functools import partial
import argparse
from PIL import Image
import torch
import copy
import numpy as np
import random
import time
from random import shuffle
import shutil
from collections import defaultdict
import os
from functools import partial

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as tv_models

from torch.utils.data import DataLoader, Dataset

from cs6740.densenet import DenseNet121

import os
import sys
import math

import shutil



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=175)
    parser.add_argument('--preTrainedImgModel', type=str, default='densenet121')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--dataRoot')
    parser.add_argument('--save')
    parser.add_argument('--preTrainedModel')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    if args.preTrainedImgModel == 'densenet121':
        net = DenseNet121(pretrained=True)
    else:
        raise Exception('only densenet recognized')

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    if True:
        testTransform = valTransform = trainTransform = None
    else:
        trainTransform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        testTransform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    print(net)
    run(args, optimizer, net, trainTransform, valTransform, testTransform)


def run(args, optimizer, net, trainTransform, valTransform, testTransform):
    data_root = args.dataRoot
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_set = dset.CocoCaptions(
        root=os.path.join(data_root, 'train2017'),
        annFile=os.path.join(data_root, 'captions_train2017.json'),
        transform=trainTransform)
    val_set = dset.CocoCaptions(
        root=os.path.join(data_root, 'val2017'),
        annFile=os.path.join(data_root, 'captions_val2017.json'),
        transform=valTransform)
    test_set = dset.CocoCaptions(
        root=os.path.join(data_root, 'test2017'),
        annFile=os.path.join(data_root, 'captions_test2017.json'),
        transform=testTransform)

    trainLoader = DataLoader(
        train_set, batch_size=args.batchSz, shuffle=True, **kwargs)
    valLoader = DataLoader(
        val_set, batch_size=args.batchSz, shuffle=False, **kwargs)
    testLoader = DataLoader(
        test_set, batch_size=args.batchSz, shuffle=False, **kwargs)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    valF = open(os.path.join(args.save, 'val.csv'), 'w')

    best_error = 100
    ts0 = time.perf_counter()
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        err = val(args, epoch, net, valLoader, optimizer, valF)

        torch.save(optimizer.state_dict(), os.path.join(args.save, 'optimizer_last_epoch.t7'))
        torch.save(net.state_dict(), os.path.join(args.save, 'model_last_epoch.t7'))

        if err < best_error:
            best_error = err
            print('New best error {}'.format(err))
            torch.save(net.state_dict(), os.path.join(args.save, 'model_best.t7'))

    trainF.close()
    valF.close()
    print('Done in {:.2f}s'.format(time.perf_counter() - ts0))


def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()

    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    ts0 = time.perf_counter()

    for batch_idx, (data, target) in enumerate(trainLoader):
        ts0_batch = time.perf_counter()

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)

        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)

        del output
        loss.backward()
        optimizer.step()
        nProcessed += len(data)

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        te = time.perf_counter()
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tTime: [{:.2f}s/{:.2f}s]\tLoss: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            te - ts0_batch, te - ts0, loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()


def val(args, epoch, net, valLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0

    ts0 = time.perf_counter()
    for data, target in valLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(valLoader) # loss function already averages over batch size
    nTotal = len(valLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Time: {:.2f}s, Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        time.perf_counter() - ts0, test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch == 1:
            lr = 1e-1
        elif epoch == 126:
            lr = 1e-2
        elif epoch == 151:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__=='__main__':
    main()
