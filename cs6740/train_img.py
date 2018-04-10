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

    def forward(self, x):
        img, txt = x
        img, txt = self.img_net(img), self.txt_net(self.text_embedding(txt))
        return torch.bmm(img.view(img.shape[0], 1, img.shape[1]), txt.view(txt.shape[0], txt.shape[1], 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=175)
    parser.add_argument('--preTrainedImgModel', type=str, default='densenet121')
    parser.add_argument('--textModel', type=str, default='bow')
    parser.add_argument('--textEmbedding', type=str, default='glove')
    parser.add_argument('--textEmbeddingSize', type=int, default=50)
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
        img_net = DenseNet121(pretrained=True, num_output_features=100)
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
        txt_net = BOWEncoder(1, args.textEmbeddingSize, 100)

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
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]),
    ])

    valTransform = testTransform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]),
    ])

    #print(net)
    run(args, optimizer, net, trainTransform, valTransform, testTransform, embedding)


def run(args, optimizer, net, trainTransform, valTransform, testTransform, embedding):
    data_root = args.dataRoot
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    rand_caption = lambda captions: embedding(captions[random.randint(0, len(captions) - 1)])
    train_set = CocoDataset(
        root=os.path.join(data_root, 'coco', 'train2017'),
        annFile=os.path.join(data_root, 'coco', 'captions_train2017.json'),
        transform=trainTransform, target_transform=rand_caption)
    val_set = CocoDataset(
        root=os.path.join(data_root, 'coco', 'val2017'),
        annFile=os.path.join(data_root, 'coco', 'captions_val2017.json'),
        transform=valTransform, target_transform=rand_caption)

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

    for batch_idx, (img, caption, labels) in enumerate(trainLoader):
        ts0_batch = time.perf_counter()
        labels = torch.FloatTensor(labels / 1.)

        if args.cuda:
            img, caption, labels = img.cuda(), caption.cuda(), labels.cuda()
        img, caption, labels = Variable(img, volatile=True), Variable(caption, volatile=True), Variable(labels)

        optimizer.zero_grad()
        output = torch.squeeze(net((img, caption)))
        print(output.shape, labels.shape)
        print(labels)
        print(output)
        loss = F.mse_loss(output, labels)

        #pred = output.data.max(1)[1] # get the index of the max log-probability
        #incorrect = pred.ne(target.data).cpu().sum()
        err = 0 #100.*incorrect/len(data)

        del output
        loss.backward()
        optimizer.step()
        nProcessed += len(labels)

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
    for data, (img, caption, labels) in valLoader:
        if args.cuda:
            img, caption, labels = img.cuda(), caption.cuda(), labels.cuda()
        img, caption, labels = Variable(img, volatile=True), Variable(caption, volatile=True), Variable(labels)

        output = net((img, caption))
        test_loss += F.HingeEmbeddingLoss(output, labels).data[0]
        #pred = output.data.max(1)[1] # get the index of the max log-probability
        #incorrect += pred.ne(target.data).cpu().sum()

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
