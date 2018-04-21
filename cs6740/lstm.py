import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

    def forward(self, in_data, lengths):
        # Max pool input word vectors
        # Assuming input of shape (batch_size, sentence_length, embedding_size)

        # Packed sequences have to be sorted
        lengths, idx = lengths.sort(descending=True)
        _, idx_rev = idx.sort(descending=False)
        in_data = in_data[idx, :, :]
        packed_input = pack_padded_sequence(in_data, lengths.cpu().numpy(), batch_first=True)

        _, (final_hidden_states, _) = self.lstm(packed_input)
        # final_hidden_states is (num_layers * num_directions, batch, hidden_size)
        if self.lstm.bidirectional:
            concatenated_states = torch.cat([final_hidden_states[0], final_hidden_states[1]], dim=1)
        else:
            concatenated_states = final_hidden_states.squeeze()
        concatenated_states = concatenated_states[idx_rev, :]

        # pool = torch.max(lstm_output, 1)[0]
        out_layer = self.final_layer(concatenated_states)

        # normalize the output layer
        norm = torch.norm(out_layer, 2, 1, keepdim=True)
        return out_layer/norm
