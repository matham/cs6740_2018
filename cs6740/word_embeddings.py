import torch.nn as nn
import torch
from torch.autograd import Variable

import numpy as np
import spacy

from torchvision.datasets import CocoCaptions as Coco

# coco_dataset = Coco(root="../val2017", annFile="../annotations/captions_val2017.json")


class WordEmbeddingUtil(object):
    def __init__(self, embedding_file="/home/matte/cs6740/data/glove.6B//glove.6B.50d.txt"):
        with open(embedding_file, 'r') as embed_file:
            embedding_lines = embed_file.readlines()
        embedding_data = [x.split() for x in embedding_lines]
        self.word_to_index = {w[0]:i for i, w in enumerate(embedding_data)}
        pretrained_weights = np.array([list(map(float, embed[1:])) for embed in embedding_data])

        self.embed = nn.Embedding(*pretrained_weights.shape)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weights))

        spacy_en = spacy.load('en')
        self.tokenizer = lambda x: [tok.text for tok in spacy_en.tokenizer(x)]

        self.embedding_dimension = pretrained_weights.shape[1]
        print("Done creating embedding matrix")

    # Tokenizes a caption and returns a tensor containing its word embeddings
    def get_embeddings(self, caption):
        if isinstance(caption, str):
            tokens = self.tokenizer(caption.lower())
            indices = [self.word_to_index[w] for w in tokens if w in self.word_to_index]
        elif isinstance(caption, list):
            # TODO: Handle different length sentences
            tokens = [self.tokenizer(c.lower()) for c in caption]
            indices = [[self.word_to_index[w] for w in cap if w in self.word_to_index] for cap in tokens]
        else:
            raise ValueError("Expected caption to be string or list")
        indices = Variable(torch.LongTensor(indices))
        return self.embed(indices)

    def __call__(self, caption):
        return self.get_embeddings(caption)



if __name__ == "__main__":
    embed_util = WordEmbeddingUtil()
    print(embed_util.get_embeddings("see jane run").shape)
