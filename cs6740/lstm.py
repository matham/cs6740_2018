import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CocoLSTM(nn.Module):

    def __init__(self, input_size, output_size):
        super(CocoLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=100,
                            num_layers=1,
                            dropout=.1,
                            batch_first=True,
                            bidirectional=True)

        self.final_layer = nn.Linear(2 * 100, output_size)

    def forward(self, in_data):
        # Max pool input word vectors
        # Assuming input of shape (batch_size, sentence_length, embedding_size)
        lstm_output = self.lstm(in_data)[0]
        pool = torch.max(lstm_output, 1)[0]
        out_layer = self.final_layer(pool)

        # normalize the output layer
        norm = torch.norm(out_layer, 2, 1, keepdim=True)
        return out_layer/norm
