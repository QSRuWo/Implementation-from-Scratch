import torch
import torch.nn as nn
import torch.nn.functional as F
from ReadData import load_data_time_machine
from tools import Timer, Accumulator, sgd
import math


batch_size, num_steps = 32, 35
train_loader, vocab = load_data_time_machine(batch_size, num_steps)

def get_params(vocab_size, hidden_size, device):
    '''
    This function is to initialize and get all parameters for training
    :param vocab_size:
    :param hidden_size:
    :param device:
    :return:
    '''
    num_inputs = num_outpus = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # These three parameters are for rnn layer, actually, there should be one more parameter b_x
    W_xh = normal((num_inputs, hidden_size))
    W_hh = normal((hidden_size, hidden_size))
    b_h = torch.zeros(hidden_size, device=device)
    # These two parameters are for output layer
    W_ho = normal((hidden_size, num_outpus))
    b_o = torch.zeros(num_outpus, device=device)
    params = [W_xh, W_hh, b_h, W_ho, b_o]
    for param in params:
        param.requires_grad = True
    return params

def init_hidden_variable(batch_size, hidden_size, device):
    '''
    This function is to define the initial hidden variable
    :param batch_size:
    :param hidden_size:
    :param device:
    :return:
    '''
    # In Pytorch, initial hidden variable should be shaped like (D * num_layers, hidden_size) or (D * num_layers, batch_size, hidden_size)
    # There is a little difference with that.
    return (torch.zeros((batch_size, hidden_size), device=device),)

def rnn_forward(inputs, state, params):
    '''
    This function is to implement a rnn forward function block
    :param inputs:
    :param state:
    :param params:
    :return:
    '''
    W_xh, W_hh, b_h, W_ho, b_o = params
    H, = state
    outputs = []
    # inputs shape is like (num_steps, batch_size, len(vocab))
    # So the X here is a two-dimension tensor
    for X in inputs:
        # The following line corresponds to rnn layer in Pytorch
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # The following line corresponds to linear layer which do the classification and follows the rnn layer in Pytorch
        Y = torch.mm(H, W_ho) + b_o
        # In pytorch, this output is a tensor which shapes like (num_steps, batch_size, len(vocab))
        outputs.append(Y)
    # The return tensor shape is (num_steps * batch_size, len(vocab))
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
    '''
    Build a RNN model class
    '''
    def __init__(self, vocab_size, hidden_size, device, get_params, init_state, forward_fn):
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        self.params = get_params(vocab_size, hidden_size, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.hidden_size, device)

hidden_size = 512
X = torch.arange(10).reshape(2,5)
device = torch.device('cuda')
net = RNNModelScratch(len(vocab), hidden_size, device, get_params, init_hidden_variable, rnn_forward)
state = net.begin_state(X.shape[0], device)
X = X.to(device)
Y, new_state = net(X, state)
print(f'Y.shape {Y.shape}')
print(f'len(new_state) {len(new_state)}')
print(f'new_state[0].shape {new_state[0].shape}')

def rnn_predict(prefix, num_preds, net, vocab, device):
    '''
    This function is to predict following text according to prefix, and the length of the following is num_preds
    :param prefix:
    :param num_preds:
    :param net:
    :param vocab:
    :param device:
    :return:
    '''
    # Here we only predict one prefix, so to simplify it, we set batch_size = 1
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda : torch.tensor(outputs[-1], device=device).reshape(1, 1)
    # We first update initial state to the state of last token of prefix
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # Then we start prediction
    for _ in range(num_preds):
        Y, state = net(get_input(), state)
        outputs.append(int(Y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# Test the prediction
print(f'Prediction test: {rnn_predict("time traveller", 10, net, vocab, device)}')

# Then define gradient clipping
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

# Define training method for one epoch
def train_one_epoch(net, train_loader, loss, optimizer, device, use_random_iter):
    '''
    This function is to define the training process in one epoch
    :param net:
    :param train_loader:
    :param loss:
    :param optimizer:
    :param device:
    :param use_random_iter:
    :return:
    '''
    state, timer = None, Timer()
    # The first position records perplexity and the second position records speed
    metric = Accumulator(2)
    for inputs, labels in train_loader:
        # The shape of inputs and labels are (batch_size, len_seq(i.e. num_seqs))
        if state is None or use_random_iter:
            # When state is None, then initialize it.
            # When using random iter, we should initialize state every time since there is no time connection between sequences
            # Which means we can not use state at t-1 to calculate state at t.
            state = net.begin_state(batch_size=inputs.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                # This branch is for Implementation from Scratch
                for s in state:
                    s.detach_()
        labels = labels.T.reshape(-1)
        inputs, labels = inputs.to(device), labels.to(device)
        preds, state = net(inputs, state)
        l = loss(preds, labels.long()).mean()
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            optimizer.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            optimizer(batch_size = 1)
        metric.add(l * labels.numel(), labels.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

# Define the whole training process
def train(net, train_loader, vocab, lr, epochs, device, use_random_iter=False):
    '''
    This function is to define a complete training process
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
        optimizer = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: rnn_predict(prefix, 50, net, vocab, device)
    for epoch in range(epochs):
        perplexity, speed = train_one_epoch(net, train_loader, loss, optimizer, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            print(f'perplexity {perplexity}')
            print(f'Speed {speed} tokens/sec on {str(device)}')

epochs, lr = 500, 1
train(net, train_loader, vocab, lr, epochs, device, use_random_iter=True)