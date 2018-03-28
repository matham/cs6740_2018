import torch.nn as nn
import torch
import numpy as np
import spacy
from torch.autograd import Variable

from torchvision.datasets import CocoCaptions as Coco

# coco_dataset = Coco(root="../val2017", annFile="../annotations/captions_val2017.json")


class WordEmbeddingUtil(object):
    def __init__(self, embedding_file=".vector_cache/glove.6B.50d.txt", dimension=50):
        with open(embedding_file, 'r') as embed_file:
            embedding_lines = embed_file.readlines()
        embedding_data = [x.split() for x in embedding_lines]
        self.word_to_index = {w[0]:i for i, w in enumerate(embedding_data)}
        pretrained_weights = np.array([list(map(float, embed[1:])) for embed in embedding_data])

        self.embed = nn.Embedding(*pretrained_weights.shape)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weights))

        spacy_en = spacy.load('en')
        self.tokenizer = lambda x: [tok.text for tok in spacy_en.tokenizer(x)]

        self.embedding_dimension = dimension
        print("Done creating embedding matrix")

    # Tokenizes a caption and returns a tensor containing its word embeddings
    def get_embeddings(self, caption):
        tokens = self.tokenizer(caption)
        print("Tokens: ", tokens)
        indices = Variable(torch.LongTensor([self.word_to_index[w] for w in tokens if w in self.word_to_index]))
        return self.embed(indices)



if __name__ == "__main__":
    embed_util = WordEmbeddingUtil()
    print(embed_util.get_embeddings("My name is Steve"))
