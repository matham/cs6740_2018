import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CocoLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(CocoLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=1000,
                            num_layers=1,
                            dropout=0,
                            batch_first=True,
                            bidirectional=True)


        self.final_layer = nn.Linear(1000, output_size)



    def forward(self, in_data):
        # Max pool input word vectors
        # Assuming input of shape (batch_size, sentence_length, embedding_size)
        #pool = torch.max(in_data, 1)[0]
        lstm_output = self.lstm(in_data)[0]
        #print(lstm_output.shape, in_data.shape, lstm_output[:, -1, :].shape)
        out_layer = self.final_layer(lstm_output[:, -1, :])
        #print(out_layer.shape, in_data.shape, lstm_output.shape, lstm_output[:, -1, :].shape)

        # normalize the output layer
        norm = torch.norm(out_layer, 2, 1, keepdim=True)
        return out_layer/norm
