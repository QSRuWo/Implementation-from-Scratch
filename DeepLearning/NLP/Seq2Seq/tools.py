import torch
import torch.nn as nn
import time

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