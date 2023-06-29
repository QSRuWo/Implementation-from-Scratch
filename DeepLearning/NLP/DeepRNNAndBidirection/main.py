import torch
import torch.nn as nn
import torch.nn.functional as F
from ReadData import load_data_time_machine
from tools import Timer, Accumulator, train
from Model import LSTMModel
import math

batch_size, num_steps = 32, 35
train_loader, vocab = load_data_time_machine(batch_size, num_steps)

vocab_size, hidden_size, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = torch.device('cuda')
lstm_layer = nn.LSTM(num_inputs, hidden_size, num_layers)
model = LSTMModel(lstm_layer, vocab_size)
model = model.to(device)
epochs, lr = 500, 2
# train(model, train_loader, vocab, lr, epochs, device)

'''
A wrong instance to use bi-direction
Bi-direction cannot be used in language model to predict next word, since there is not 'future' info in the input.
'''
lstm_layer = nn.LSTM(num_inputs, hidden_size, num_layers, bidirectional=True)
model = LSTMModel(lstm_layer, vocab_size)
model = model.to(device)
epochs, lr = 500, 2
train(model, train_loader, vocab, lr, epochs, device)