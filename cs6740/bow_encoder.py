import torch.nn as nn

class BOWEncoder(nn.Module):
    def __init__(self, layer_size, input_size, output_size):
        self.layer_size = layer_size

        self.layer_one = nn.linear(input_size, layer_size)
        self.layer_two = nn.linear(layer_size, output_size)
        self.activation = nn.ReLU()

    def encode(self, in_data):
        l1 = self.activation(self.layer_one(in_data))
        l2 = self.activation(self.layer_two(l1))
        return l2
