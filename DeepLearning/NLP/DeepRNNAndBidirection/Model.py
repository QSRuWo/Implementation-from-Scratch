import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, lstm_layer, vocab_size, **kwargs):
        super(LSTMModel, self).__init__()
        self.lstm = lstm_layer
        self.vocab_size = vocab_size
        self.hidden_size = lstm_layer.hidden_size
        if not self.lstm.bidirectional:
            self.num_direction = 1
            self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        else:
            self.num_direction = 2
            self.linear = nn.Linear(self.hidden_size * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.lstm(X, state)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.lstm, nn.LSTM):
            return torch.zeros((self.num_direction * self.lstm.num_layers, batch_size, self.hidden_size), device=device)
        else:
            return (torch.zeros((self.num_direction * self.lstm.num_layers, batch_size, self.hidden_size), device=device),
                    torch.zeros((self.num_direction * self.lstm.num_layers, batch_size, self.hidden_size), device=device))