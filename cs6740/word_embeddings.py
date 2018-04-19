import torch.nn as nn
import torch
from torch.autograd import Variable

import numpy as np
import spacy

from torchvision.datasets import CocoCaptions as Coco


class WordEmbeddingUtil(object):
    def __init__(self, padding=102, embedding_file="data/glove.6B/glove.6B.50d.txt"):
        self.padding = padding
        try:
            with open(embedding_file[:-3] + 'npz', 'rb') as embed_file:
                npzfile = np.load(embed_file)
                pretrained_weights, vocab = [npzfile[f] for f in npzfile.files]
            self.word_to_index = {w: i for i, w in enumerate(vocab)}
        except OSError:
            vocab = []
            with open(embedding_file, 'r') as embed_file:
                for line in embed_file:
                    embed = line.split()
                    vocab.append(embed[0])
                    self.embedding_dimension = len(embed) - 1

            self.word_to_index = {w: i for i, w in enumerate(vocab)}
            pretrained_weights = np.loadtxt(embedding_file, usecols=range(1, self.embedding_dimension+1), delimiter=' ', comments=None)
            pretrained_weights = np.append(pretrained_weights, np.zeros([1, self.embedding_dimension]), axis=0)
            np.savez(embedding_file[:-3] + 'npz', pretrained_weights, vocab)

        self.embed = nn.Embedding(*pretrained_weights.shape, padding_idx=len(self.word_to_index))
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weights))

        spacy_en = spacy.load('en')
        self.tokenizer = lambda x: [tok.text for tok in spacy_en.tokenizer(x)]

        self.embedding_dimension = pretrained_weights.shape[1]

    # Tokenizes a caption and returns a tensor containing its word embeddings
    def get_embeddings(self, caption):
        if isinstance(caption, str):
            tokens = self.tokenizer(caption.lower())
            true_length = len(tokens)
            indices = [self.word_to_index[w] for w in tokens if w in self.word_to_index]
            indices += [len(self.word_to_index), ] * (self.padding - len(indices))

        indices = torch.LongTensor(indices)
        return indices, true_length

    def __call__(self, caption):
        return self.get_embeddings(caption)



if __name__ == "__main__":
    embed_util = WordEmbeddingUtil()
    print(embed_util.get_embeddings("see jane run").shape)
