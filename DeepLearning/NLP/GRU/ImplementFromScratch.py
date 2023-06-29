import torch
import torch.nn as nn
import torch.nn.functional as F
from ReadData import load_data_time_machine
from tools import Timer, Accumulator, sgd
import math

# First get the time machine data
batch_size, num_steps = 32, 35
train_loader, vocab = load_data_time_machine(batch_size, num_steps)

# Initialize the parameters of layer
def get_params(vocab_size, hidden_size, device):
    '''
    This function is to initialize parameters of GRU layer.
    :param vocab_size:
    :param hidden_size:
    :param device:
    :return:
    '''
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(shape, device=device) * 0.01

    def three():
        return (
            normal((num_inputs, hidden_size)),
            normal((hidden_size, hidden_size)),
            torch.zeros(hidden_size, device=device)
        )

    # Update gate
    W_xz, W_hz, b_z = three()
    # Reset gate
    W_xr, W_hr, b_r = three()
    # Candidate hidden state
    W_xh, W_hh, b_h = three()
    # Calculate new state
    W_ho = normal((hidden_size, num_outputs))
    b_o = torch.zeros(num_outputs, device=device)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_ho, b_o]
    for param in params:
        param.requires_grad = True
    return params

def init_hidden_state(batch_size, hidden_size, device):
    '''
    Initialize the first hidden state
    :param batch_size:
    :param hidden_size:
    :param device:
    :return:
    '''
    return (torch.zeros((batch_size, hidden_size), device=device),)

def gru_forward_function(inputs, state, params):
    '''
    This function is to implement the forward function of GRU model
    :param inputs:
    :param state:
    :param params:
    :return:
    '''
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_ho, b_o = params
    H, = state
    outputs = []
    for x in inputs:
        # Calculate Update gate
        Z = torch.sigmoid(torch.mm(x, W_xz) + torch.mm(H, W_hz) + b_z)
        # Calculate Reset gate
        R = torch.sigmoid(torch.mm(x, W_xr) + torch.mm(H, W_hr) + b_r)
        # Calculate Candidate hidden state
        H_candidate = torch.tanh(torch.mm(x, W_xh) + torch.mm(R * H, W_hh) + b_h)
        # Calculate new state
        H = Z * H + (1 - Z) * H_candidate
        # Calculate output
        y = torch.mm(H, W_ho) + b_o
        outputs.append(y)
    # The first return shape is like (num_steps * batch_size, len(vocab))
    return torch.cat(outputs, dim=0), (H,)

class GRUModelScratch:
    '''
    Define GRU model
    '''
    def __init__(self, vocab_size, hidden_size, device, get_params, init_state, forward_fn):
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        self.params = get_params(vocab_size, hidden_size, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, inputs, state):
        inputs = F.one_hot(inputs.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(inputs, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.hidden_size, device)

def gru_predict(prefix, num_steps, net, vocab, device):
    '''
    This function is to use gru to do prediction according to prefix.
    :param prefix:
    :param num_steps:
    :param net:
    :param vocab:
    :param device:
    :return:
    '''
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda : torch.tensor(outputs[-1], device=device).reshape(1, 1)
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_steps):
        Y, state = net(get_input(), state)
        pred = int(Y.argmax(dim=1).reshape(1))
        outputs.append(pred)
    return ''.join([vocab.idx_to_token[idx] for idx in outputs])

def grad_clipping(net, theta):
    '''
    This function is to implement gradient clipping
    :param net:
    :param theta:
    :return:
    '''
    if isinstance(net, nn.Module):
        params = [param for param in net.parameters() if param.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((param.grad ** 2)) for param in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_one_epoch(net, train_loader, loss, optimizer, device, use_random_iter):
    '''
    This function is to define the process of training in one epoch
    :param net:
    :param train_loader:
    :param loss:
    :param optimizer:
    :param device:
    :param use_random_iter:
    :return:
    '''
    state, timer = None, Timer()
    metric = Accumulator(2)
    for inputs, labels in train_loader:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=inputs.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        labels = labels.T.reshape(-1)
        inputs, labels = inputs.to(device), labels.to(device)
        Y, state = net(inputs, state)
        l = loss(Y, labels.long()).mean()
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
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train(net, train_loader, vocab, lr, epochs, device, use_random_iter=False):
    '''
    Define a complete process of training
    :param net:
    :param train_loader:
    :param vocab:
    :param lr:
    :param epochs:
    :param device:
    :param use_random_iter:
    :return:
    '''
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        optimizer = lambda batch_size: sgd(net.params, lr=lr, batch_size=batch_size)
    predict = lambda prefix: gru_predict(prefix, 50, net, vocab, device)
    for epoch in range(epochs):
        perplexity, speed = train_one_epoch(net, train_loader, loss, optimizer, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}')
            print(predict('time traveller'))
            print(f'perplexity {perplexity}')
            print(f'speed {speed} tokens/sec on {str(device)}')

vocab_size, hidden_size, device = len(vocab), 256, torch.device('cuda')
epochs, lr = 500, 1
net = GRUModelScratch(vocab_size, hidden_size, device, get_params, init_hidden_state, gru_forward_function)
# train(net, train_loader, vocab, lr, epochs,device)

'''
    Concise implementation
'''

class GRUModel(nn.Module):
    def __init__(self, gru_layer, vocab_size, **kwargs):
        super(GRUModel, self).__init__()
        self.gru = gru_layer
        self.vocab_size = vocab_size
        self.hidden_size = gru_layer.hidden_size
        if not self.gru.bidirectional:
            self.num_direction = 1
            self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        else:
            self.num_direction = 2
            self.linear = nn.Linear(self.hidden_size * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.gru(X, state)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.gru, nn.LSTM):
            return torch.zeros((self.num_direction * self.gru.num_layers, batch_size, self.hidden_size), device=device)
        else:
            return (torch.zeros((self.num_direction * self.gru.num_layers, batch_size, self.hidden_size), device=device),
                    torch.zeros((self.num_direction * self.gru.num_layers, batch_size, self.hidden_size), device=device))

num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, hidden_size)
model = GRUModel(gru_layer, vocab_size)
model = model.to(device)
train(model, train_loader, vocab, lr, epochs, device)