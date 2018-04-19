import torch.nn as nn
import torch


class BOWEncoder(nn.Module):
    def __init__(self, layer_size, input_size, output_size):
        super(BOWEncoder, self).__init__()
        self.layer_size = layer_size

        self.hidden_layer = nn.Linear(input_size, output_size)

    def forward(self, in_data, lengths):
        # Max pool input word vectors
        # Assuming input of shape (batch_size, sentence_length, embedding_size)
        pool = torch.max(in_data, 1)[0]
        out_layer = self.hidden_layer(pool)
        # normalize the output layer
        norm = torch.norm(out_layer, 2, 1, keepdim=True)
        return out_layer/norm
