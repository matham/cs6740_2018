import torch.nn as nn
import torch

class BOWEncoder(nn.Module):
    def __init__(self, layer_size, input_size, output_size):
        super(BOWEncoder, self).__init__()
        self.layer_size = layer_size

        self.layer_one = nn.linear(input_size, layer_size)
        self.layer_two = nn.linear(layer_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, in_data):
        # Max pool input word vectors
        # Assuming input of shape (batch_size, sentence_length, embedding_size)
        pool = torch.max(in_data, 1)
        l1 = self.activation(self.layer_one(pool))
        l2 = self.layer_two(l1)
        # normalize the output layer
        norm = torch.norm(l2, 2, 1, keep_dim=True)
        return l2/norm
