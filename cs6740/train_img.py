#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import random
import time

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from cs6740.densenet import DenseNet121
from cs6740.lstm import LSTM
from cs6740.bow_encoder import BOWEncoder
from cs6740.data_loaders import CocoDataset
from cs6740.word_embeddings import WordEmbeddingUtil

import os
import sys
import math

import shutil


class FinalLayer(nn.Module):

    def __init__(self, img_net, txt_net, text_embedding, **kwargs):
        super(FinalLayer, self).__init__(**kwargs)
        self.img_net = img_net
        self.txt_net = txt_net
        self.text_embedding = text_embedding.embed
        for param in self.text_embedding.parameters():
            param.requires_grad = False

    def forward(self, x):
        img, txt = x
        img = self.img_net(img)
        txt = self.text_embedding(txt)
        if self.txt_net is not None:
            txt = self.txt_net(txt)
        return torch.squeeze(img), torch.squeeze(txt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=175)
    parser.add_argument('--preTrainedImgModel', type=str, default='densenet121')
    parser.add_argument('--textModel', type=str, default='bow')
    parser.add_argument('--textEmbedding', type=str, default='glove')
    parser.add_argument('--textEmbeddingSize', type=int, default=50)
    parser.add_argument('--valSubset', type=str, default='')
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
        img_net = DenseNet121(pretrained=True, num_output_features=1000)
        for param in img_net.features.parameters():
            param.requires_grad = False
    else:
        raise Exception('only densenet recognized')

    if args.textEmbedding == 'glove':
        embedding = WordEmbeddingUtil(
            embedding_file=os.path.join(
                args.dataRoot, 'glove.6B',
                'glove.6B.{}d.txt'.format(args.textEmbeddingSize)))

    if args.textModel == 'bow':
        txt_net = BOWEncoder(1, args.textEmbeddingSize, 1000)

    net = FinalLayer(img_net=img_net, txt_net=txt_net, text_embedding=embedding)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=1e-4)

    trainTransform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]),
    ])

    valTransform = testTransform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]),
    ])

    #print(net)
    run(args, optimizer, net, trainTransform, valTransform, testTransform, embedding)


def run(args, optimizer, net, trainTransform, valTransform, testTransform, embedding):
    data_root = args.dataRoot
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    ranks = [1, 3, 5, 10, 100000]

    rand_caption = lambda captions: embedding(captions[random.randint(0, len(captions) - 1)])
    subset = args.valSubset or None
    if subset:
        with open(subset, 'r') as fh:
            subset = list(map(int, fh.read().split(',')))

    train_set = CocoDataset(
        root=os.path.join(data_root, 'coco', 'train2017'),
        annFile=os.path.join(data_root, 'coco', 'captions_train2017.json'),
        transform=trainTransform, target_transform=rand_caption)
    val_set = CocoDataset(
        root=os.path.join(data_root, 'coco', 'val2017'),
        annFile=os.path.join(data_root, 'coco', 'captions_val2017.json'),
        transform=valTransform, target_transform=rand_caption, subset=subset)

    trainLoader = DataLoader(
        train_set, batch_size=args.batchSz, shuffle=True, **kwargs)
    valLoader = DataLoader(
        val_set, batch_size=args.batchSz, shuffle=False, **kwargs)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    valF = open(os.path.join(args.save, 'val.csv'), 'w')

    best_error = 100
    ts0 = time.perf_counter()
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF, ranks)
        err = val(args, epoch, net, valLoader, optimizer, valF, ranks)

        torch.save(optimizer.state_dict(), os.path.join(args.save, 'optimizer_last_epoch.t7'))
        torch.save(net.state_dict(), os.path.join(args.save, 'model_last_epoch.t7'))

        if err < best_error:
            best_error = err
            print('New best error {}'.format(err))
            torch.save(net.state_dict(), os.path.join(args.save, 'model_best.t7'))

    trainF.close()
    valF.close()
    print('Done in {:.2f}s'.format(time.perf_counter() - ts0))


def compute_ranking(texts, images, labels, ranks=[1]):
    texts, images, labels = texts.cpu().data, images.cpu().data, labels.cpu().data

    batch_match = torch.arange(texts.shape[0]).long()[labels == 1]
    n = batch_match.shape[0] if batch_match.shape else 0
    if not n:
        return [0 for _ in ranks], 0

    texts = texts.index_select(0, batch_match)
    images = images.index_select(0, batch_match)
    similarity = torch.mm(texts, images.transpose(1, 0))
    sort_val, sort_key = torch.sort(similarity, dim=1, descending=True)
    labels = torch.arange(n).long().expand(n, n).transpose(1, 0)
    matched = labels == sort_key

    accuracy = [matched[:, :rank].sum() / n * 100 for rank in ranks]
    return accuracy, n


def train(args, epoch, net, trainLoader, optimizer, trainF, ranks):
    net.train()

    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    ts0 = time.perf_counter()

    for batch_idx, (img, caption, labels) in enumerate(trainLoader):
        ts0_batch = time.perf_counter()
        labels = labels.float()

        if args.cuda:
            img, caption, labels = img.cuda(), caption.cuda(), labels.cuda()
        img, caption, labels = Variable(img), Variable(caption), Variable(labels)

        optimizer.zero_grad()
        output = net((img, caption))
        rankings, _ = compute_ranking(*output[::-1], labels, ranks)
        loss = F.cosine_embedding_loss(*output, labels)

        #pred = output.data.max(1)[1] # get the index of the max log-probability
        #incorrect = pred.ne(target.data).cpu().sum()
        err = 0 #100.*incorrect/len(data)

        del output
        loss.backward()
        optimizer.step()
        nProcessed += len(labels)

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        te = time.perf_counter()
        s = 'Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tTime: [{:.2f}s/{:.2f}s]\tLoss: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            te - ts0_batch, te - ts0, loss.data[0], err)
        s_rank = '\t'.join((['{:.2f}', ] * len(rankings))).format(*rankings)
        print(s + '\tRanks: ' + s_rank)

        trainF.write('{},{},{},{}\n'.format(
            partialEpoch, loss.data[0], err,
            ','.join((['{}', ] * len(rankings))).format(*rankings)))
        trainF.flush()


def val(args, epoch, net, valLoader, optimizer, testF, ranks):
    net.eval()
    test_loss = 0
    incorrect = 0
    rank_values = [0, ] * len(ranks)
    num_ranks = 0

    ts0 = time.perf_counter()
    for batch_idx, (img, caption, labels) in enumerate(valLoader):
        labels = labels.float()
        if args.cuda:
            img, caption, labels = img.cuda(), caption.cuda(), labels.cuda()
        img, caption, labels = Variable(img), Variable(caption), Variable(labels)

        output = net((img, caption))
        rankings, rank_count = compute_ranking(*output[::-1], labels, ranks)
        if rank_count:
            for i, value in enumerate(rankings):
                rank_values[i] += value
            num_ranks += 1
        test_loss += F.cosine_embedding_loss(*output, labels).data[0]
        #pred = output.data.max(1)[1] # get the index of the max log-probability
        #incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(valLoader) # loss function already averages over batch size
    nTotal = len(valLoader.dataset)
    err = 100.*incorrect/nTotal
    s = '\nTest set: Time: {:.2f}s\tAverage loss: {:.4f}\tError: {}/{} ({:.0f}%)'.format(
        time.perf_counter() - ts0, test_loss, incorrect, nTotal, err)

    rankings = [r / num_ranks for r in rank_values]
    s_rank = '\t'.join((['{:.2f}', ] * len(rankings))).format(*rankings)
    print(s + '\tRanks: ' + s_rank + '\n')

    testF.write('{},{},{},{}\n'.format(
        epoch, test_loss, err,
        ','.join((['{}', ] * len(rankings))).format(*rankings)))
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
