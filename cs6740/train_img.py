#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import argparse
import numpy as np
import random
import time
import itertools

from PIL import Image
import torchvision.datasets as dset
import matplotlib.pyplot as plt

from cs6740.densenet import DenseNet121
from cs6740.lstm import CocoLSTM
from cs6740.bow_encoder import BOWEncoder
from cs6740.data_loaders import CocoDataset, CocoDatasetConstSize
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
        img, txt, lengths = x
        img = self.img_net(img)
        txt = self.text_embedding(txt)
        if self.txt_net is not None:
            txt = self.txt_net(txt, lengths)
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
    parser.add_argument('--babyCoco', action='store_true')
    parser.add_argument('--testOnly', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--dataRoot')
    parser.add_argument('--save')
    parser.add_argument('--preTrainedModel', type=str, default='')
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
    elif args.textModel == 'lstm':
        txt_net = CocoLSTM(args.textEmbeddingSize, 1000)

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
    tboard_writer = SummaryWriter(log_dir=args.save)

    if args.testOnly:
        run_test(args, net, valTransform, embedding)
    elif args.demo:
        score_images_on_caption(args, net, embedding, valTransform)
    else:
        run(args, optimizer, net, trainTransform, valTransform, testTransform, embedding, tboard_writer)


def run_test(args, net, valTransform, embedding):
    data_root = args.dataRoot
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    ranks = [1, 3, 5, 10, 100000]

    rand_caption = lambda captions: embedding(captions[random.randint(0, len(captions) - 1)])
    subset = args.valSubset or None
    if subset:
        with open(subset, 'r') as fh:
            subset = list(map(int, fh.read().split(',')))
    if args.babyCoco:
        subset = list(range(32))

    val_set = CocoDataset(
        root=os.path.join(data_root, 'coco', 'val2017'),
        annFile=os.path.join(data_root, 'coco', 'captions_val2017.json'),
        transform=valTransform, target_transform=rand_caption, subset=subset)

    valLoader = DataLoader(
        val_set, batch_size=args.batchSz, shuffle=False, **kwargs)

    valF = open(os.path.join(args.save, 'val_test_only.csv'), 'w')

    ts0 = time.perf_counter()

    if args.preTrainedModel:
        state = net.state_dict()
        state.update(torch.load(args.preTrainedModel))
        net.load_state_dict(state)
        del state

    err, rank_vals_all = val(args, 0, net, valLoader, None, valF, ranks, None)
    rank_vals_all = torch.cat(rank_vals_all).numpy()
    np.savetxt(os.path.join(args.save, 'val_ranks.csv'), rank_vals_all, delimiter=',')

    valF.close()
    print('Done in {:.2f}s'.format(time.perf_counter() - ts0))


def run(args, optimizer, net, trainTransform, valTransform, testTransform, embedding, tboard_writer):
    data_root = args.dataRoot
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    ranks = [1, 3, 5, 10, 100000]

    rand_caption = lambda captions: embedding(captions[random.randint(0, len(captions) - 1)])
    subset = args.valSubset or None
    train_subset = None
    if subset:
        with open(subset, 'r') as fh:
            subset = list(map(int, fh.read().split(',')))
    if args.babyCoco:
        subset = train_subset = list(range(32))

    train_set = CocoDatasetConstSize(
        root=os.path.join(data_root, 'coco', 'train2017'),
        annFile=os.path.join(data_root, 'coco', 'captions_train2017.json'),
        transform=trainTransform, target_transform=rand_caption, subset=train_subset,
        proportion_positive=.2)
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
        if epoch == 3:
            train_set.proportion_positive = .1
        # elif epoch == 5:
        #     train_set.proportion_positive = .05
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF, ranks, tboard_writer)
        err, _ = val(args, epoch, net, valLoader, optimizer, valF, ranks, tboard_writer)

        torch.save(optimizer.state_dict(), os.path.join(args.save, 'optimizer_last_epoch.t7'))
        torch.save(net.state_dict(), os.path.join(args.save, 'model_last_epoch.t7'))

        if err < best_error:
            best_error = err
            print('New best error {}'.format(err))
            torch.save(net.state_dict(), os.path.join(args.save, 'model_best.t7'))

    trainF.close()
    valF.close()
    print('Done in {:.2f}s'.format(time.perf_counter() - ts0))


def compute_ranking(texts, images, labels, img_indices, tboard_writer, ranks=[1]):
    texts, images, labels = texts.cpu().data, images.cpu().data, labels.cpu().data
    # _, unique_indices = np.unique(img_indices.numpy(), return_index=True)

    batch_match = torch.arange(texts.shape[0]).long()[labels == 1]
    n = batch_match.shape[0] if batch_match.shape else 0
    if not n:
        return [0 for _ in ranks], 0, None, 0, None

    texts = texts.index_select(0, batch_match)
    images = images.index_select(0, batch_match)
    # for each caption, compute its similarity with every positive image
    similarity = torch.mm(texts, images.transpose(1, 0))
    # sort by similarity so each caption has a list of images from most to
    # least similar
    sort_val, sort_key = torch.sort(similarity, dim=1, descending=True)
    # for each caption, get the index of the matching image, replicate it
    # n times and then set the index in the sort key that is this index to
    # one and zero otherwise (there should be only one 1)
    labels = torch.arange(n).long().expand(n, n).transpose(1, 0)
    matched = labels == sort_key
    rank_vals = torch.arange(n).expand(n, n)[matched]
    mean_rank = torch.mean(rank_vals)

    accuracy = [matched[:, :rank].sum() / n * 100 for rank in ranks]
    return accuracy, n, similarity, mean_rank, rank_vals


def train(args, epoch, net, trainLoader, optimizer, trainF, ranks, tboard_writer):
    net.train()

    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    ts0 = time.perf_counter()

    for batch_idx, (img, (caption, lengths), labels, img_indices) in enumerate(trainLoader):
        ts0_batch = time.perf_counter()
        labels = labels.float()

        if args.cuda:
            img, caption, labels, lengths = img.cuda(), caption.cuda(), labels.cuda(), lengths.cuda()
        img, caption, labels = Variable(img), Variable(caption), Variable(labels)

        optimizer.zero_grad()
        output = net((img, caption, lengths))
        rankings, rank_count, similarity, mean_rank, rank_vals = compute_ranking(*output[::-1], labels, img_indices, tboard_writer, ranks)
        loss = F.cosine_embedding_loss(*output, labels)

        #pred = output.data.max(1)[1] # get the index of the max log-probability
        #incorrect = pred.ne(target.data).cpu().sum()
        err = 0 #100.*incorrect/len(data)

        del output
        loss.backward()
        optimizer.step()
        nProcessed += len(labels)

        if rank_count:
            expected_ranks = [min(100., r / rank_count * 100) for r in ranks]
            mean_rank_prop = mean_rank / rank_count * 100
        else:
            expected_ranks = [0, ] * len(ranks)
            mean_rank_prop = 0

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        te = time.perf_counter()
        s = 'Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tTime: [{:.2f}s/{:.2f}s]\tLoss: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            te - ts0_batch, te - ts0, loss.data[0], err)
        s_rank = '\t{:.2f}\t'.format(mean_rank)
        s_rank += '\t'.join((['{:.2f}', ] * len(rankings))).format(*rankings)
        print(s + '\tRanks: ' + s_rank)

        _rankings = list(itertools.chain(*zip(rankings, expected_ranks)))
        trainF.write('{},{},{},{},{},{}\n'.format(
            partialEpoch, loss.data[0], err, mean_rank, mean_rank_prop,
            ','.join((['{}', ] * len(_rankings))).format(*_rankings)))
        trainF.flush()

        global_step = epoch*(batch_idx+1)*partialEpoch*len(trainLoader)
        tboard_writer.add_scalar('train/loss', loss.data[0], global_step)
        for rank, n in zip(rankings, ranks):
            tboard_writer.add_scalar('train/Percent Accuracy (top {})'.format(n), rank, global_step)
        tboard_writer.add_scalar('train/Mean rank', mean_rank, global_step)
        tboard_writer.add_scalar('train/Mean rank percent', mean_rank_prop, global_step)
        # if rank_vals is not None:
        #     tboard_writer.add_histogram("train/rank_vals", rank_vals.numpy(), global_step, bins="auto")


def val(args, epoch, net, valLoader, optimizer, testF, ranks, tboard_writer):
    net.eval()
    test_loss = 0
    incorrect = 0
    rank_values = [0, ] * (2 * len(ranks))
    num_ranks = 0
    mean_rank_total = 0
    rank_vals_all = []

    ts0 = time.perf_counter()
    for batch_idx, (img, (caption, lengths), labels, img_indices) in enumerate(valLoader):
        labels = labels.float()
        if args.cuda:
            img, caption, labels, lengths = img.cuda(), caption.cuda(), labels.cuda(), lengths.cuda()
        img, caption, labels = Variable(img), Variable(caption), Variable(labels)

        output = net((img, caption, lengths))
        rankings, rank_count, similarity, mean_rank, rank_vals = compute_ranking(*output[::-1], labels, img_indices, tboard_writer, ranks)
        if rank_count:
            rank_vals_all.append(rank_vals)
            mean_rank_total += mean_rank
            for i, value in enumerate(rankings):
                rank_values[2 * i] += value
                rank_values[2 * i + 1] += min(100., ranks[i] / rank_count * 100)
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
    s_rank = '\t{:.2f}\t'.format(mean_rank_total / num_ranks)
    s_rank += '\t'.join((['{:.2f}', ] * len(rankings))).format(*rankings)
    print(s + '\tRanks: ' + s_rank + '\n')

    testF.write('{},{},{},{},{}\n'.format(
        epoch, test_loss, err, mean_rank_total / num_ranks,
        ','.join((['{}', ] * len(rankings))).format(*rankings)))
    testF.flush()

    if tboard_writer is not None:
        tboard_writer.add_scalar('val/loss', test_loss, epoch)
        for rank, n in zip(rankings[::2], ranks):
            tboard_writer.add_scalar('val/Percent Accuracy (top {})'.format(n), rank, epoch)
        tboard_writer.add_scalar('val/Mean rank', mean_rank_total / num_ranks, epoch)

        # if rank_vals_all:
        #    tboard_writer.add_histogram("val/rank_vals", torch.cat(rank_vals_all).numpy(), epoch, bins="auto")

    return err, rank_vals_all


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch in list(range(1, 9)):
            lr = 1e-1
        elif epoch in list(range(9, 17)):
            lr = 1e-2
        elif epoch in list(range(17, 21)):
            lr = 1e-3
        elif epoch in list(range(21, 25)):
            lr = 1e-4
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def score_images_on_caption(args, net, embedding, valTransform):
    net.eval()
    if not args.babyCoco:
        state = net.state_dict()
        state.update(torch.load(args.preTrainedModel))
        net.load_state_dict(state)
        del state

    rand_caption = lambda captions: embedding(captions[random.randint(0, len(captions) - 1)])
    subset = args.valSubset or None
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if subset:
        with open(subset, 'r') as fh:
            subset = list(map(int, fh.read().split(',')))
    if args.babyCoco:
        subset = list(range(32))

    val_set = CocoDataset(
        root=os.path.join(args.dataRoot, 'coco', 'val2017'),
        annFile=os.path.join(args.dataRoot, 'coco', 'captions_val2017.json'),
        transform=valTransform, target_transform=rand_caption, subset=subset)

    valLoader = DataLoader(
        val_set, batch_size=args.batchSz, shuffle=False, **kwargs)

    cos = torch.nn.CosineSimilarity(dim=1)

    coco = dset.CocoCaptions(root=os.path.join(args.dataRoot, 'coco', 'val2017'),
                             annFile=os.path.join(args.dataRoot, 'coco', 'captions_val2017.json'))

    while True:
        caption = input("Caption: ").strip()
        caption, length = embedding(caption)
        if args.cuda:
            caption = caption.cuda()
        caption = Variable(caption)
        length = torch.from_numpy(np.array([length]))
        result = {}


        for batch_idx, (img, (_, _), _, img_indices) in enumerate(valLoader):
            if args.cuda:
                img = img.cuda()
            img = Variable(img)
            batched_captions = caption.expand(img.shape[0], -1)
            batched_lengths = torch.squeeze(length.expand(img.shape[0], -1))

            img_out, txt_out = net((img, batched_captions, batched_lengths))
            print("out shapes", img_out.shape, txt_out.shape)
            output = cos(img_out, txt_out)
            print("cosine shape", output.shape)
            for score, index in zip(output, img_indices):
                print(score, index)
                result[index] = float(score.data)


        fig=plt.figure(figsize=(8, 8))
        for index, (i, score) in enumerate(sorted(result.items(), key=lambda x: x[1], reverse=True)[:5]):
            img, _ = coco[i]
            fig.add_subplot(2, 5, index+1)
            plt.imshow(img)
            plt.xlabel(str(score))
            print(i, score)
        plt.show()
        input()




if __name__=='__main__':
    main()
