import torch
import torch.nn as nn
import time
import math

def sgd(params, lr, batch_size):
    '''
    Implement a simplified SGD optimizer
    :param params:
    :param lr:
    :param batch_size:
    :return:
    '''
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def grad_clipping(net, theta):
    '''
    This function is to clip gradients for avoiding gradient boosting
    :param net:
    :param theta:
    :return:
    '''
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        # This is for implementation from scratch
        params = net.params
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class Timer:
    '''
    For recording running time
    '''
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

class Accumulator:
    '''
    For accumulating sums over n variables
    '''
    def __init__(self, n):
        self.data = [0.] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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