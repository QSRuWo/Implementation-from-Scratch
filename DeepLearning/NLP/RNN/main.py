import torch
import torch.nn as nn
import torch.nn.functional as F
from ReadData import load_data_time_machine
from tools import sgd, Timer, Accumulator, grad_clipping
import math

batch_size, num_step = 32, 35
# Load data time machine
train_loader, vocab = load_data_time_machine(batch_size, num_step)

print(f'len(vocab) {len(vocab)}')
# Define RNN net
hidden_size = 256
rnn_layer = nn.RNN(len(vocab), hidden_size)

# Initialize hidden state
state = torch.zeros((1, batch_size, hidden_size))
print(f'state.shape {state.shape}')

# A simple test of how to use RNN net
X = torch.rand((num_step, batch_size, len(vocab)))
print(f'X.shape {X.shape}')
Y, state_new = rnn_layer(X, state)
print(f'Y.shape {Y.shape}')
print(f'state_new.shape {state_new.shape}')

# Define a complete class for RNN model
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.hidden_size = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.hidden_size * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        # X.shape is (num_steps, batch_size, len(vocab))
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # Y.shape is (num_steps, batch_size, hidden_size if not bidirectional else hidden_size * 2)
        # state.shape is (1, batch_size, hidden_size if not bidirectional else hidden_size * 2)
        # output shape is (num_steps * batch_size, len(vocab))
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.hidden_size), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.hidden_size), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.hidden_size), device=device))

# Simple predict test before training
device = torch.device('cuda')
net = RNNModel(rnn_layer, len(vocab))
net = net.to(device)

def rnn_predict(prefix, num_preds, net, vocab, device):
    '''
    This function is to predict text according to prefix by rnn
    :param prefix:
    :param num_preds:
    :param net:
    :param vocab:
    :param device:
    :return:
    '''
    state = net.begin_state(device, batch_size=1)
    output = [vocab[prefix[0]]]
    get_input = lambda : torch.reshape(torch.tensor([output[-1]], device=device), (1,1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        output.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        output.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in output])

pred = rnn_predict('time traveller', 10, net, vocab, device)
print(pred)

# Start Training
epochs, lr = 500, 1

def rnn_train(net, train_loader, vocab, lr, epochs, device, use_random_iter=False):
    '''
    This function is to train rnn model
    :param net:
    :param train_loader:
    :param vocab:
    :param lr:
    :param epochs:
    :param device:
    :param use_random_iter:
    :return:
    '''
    # In essence, text prediction is actually a classification task
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        # This is for implementing from scratch
        optimizer = lambda batch_size : sgd(net.parameters(), lr=lr, batch_size=batch_size)
    predict = lambda prefix: rnn_predict(prefix, 50, net, vocab, device)
    for epoch in range(epochs):
        state, timer = None, Timer()
        metric = Accumulator(2)
        for inputs, labels in train_loader:
            # Here inputs, labels shape are (batch_size, sequence length)
            if state is None or use_random_iter:
                state = net.begin_state(batch_size=inputs.shape[0], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # State is a tensor for nn.GRU
                    state.detach_()
                else:
                    # State is a tuple of tensors for nn.LSTM and for custom scratch implementation
                    for s in state:
                        s.detach_()
            labels = labels.T.reshape(-1)
            inputs, labels = inputs.to(device), labels.to(device)
            print(f'inputs shape {inputs.shape}')
            print(f'labels shape {labels.shape}')
            print(f'state shape {state.shape}')
            output, state = net(inputs, state)
            print(f'output shape {output.shape}')
            l = loss(output, labels.long()).mean()
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.zero_grad()
                l.backward()
                grad_clipping(net, 1)
                optimizer.step()
            else:
                l.backward()
                grad_clipping(net, 1)
                optimizer(batch_size=1)
            metric.add(l * labels.numel(), labels.numel())
        perplexity, speed = math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
        print(f'Epoch {epoch}, Perplexity {perplexity:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))

rnn_train(net, train_loader, vocab, lr, epochs, device)