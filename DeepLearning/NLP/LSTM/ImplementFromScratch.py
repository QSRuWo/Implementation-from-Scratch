import torch
import torch.nn as nn
import torch.nn.functional as F
from ReadData import load_data_time_machine
from tools import Timer, Accumulator, sgd
import math

batch_size, num_steps = 32, 35
train_loader, vocab = load_data_time_machine(batch_size, num_steps)

def get_lstm_params(vocab_size, hidden_size, device):
    '''
    This function is to define parameters of LSTM layer
    :param vocab_size:
    :param hidden_size:
    :param device:
    :return:
    '''
    num_inputs = num_outpus = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (
            normal((num_inputs, hidden_size)),
            normal((hidden_size, hidden_size)),
            torch.zeros((hidden_size,), device=device)
        )

    # Input gate
    W_xi, W_hi, b_i = three()
    # Forget gate
    W_xf, W_hf, b_f = three()
    # Output gate
    W_xo, W_ho, b_o = three()
    # Candidate memory
    W_xc, W_hc, b_c = three()
    # Predict
    W_hp = normal((hidden_size, num_outpus))
    b_p = torch.zeros((num_outpus,), device=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hp, b_p]
    for param in params:
        param.requires_grad = True
    return params

def init_lstm_state_memory(batch_size, hidden_size, device):
    '''
    This function is to initialize the state and memory of LSTM
    :param batch_size:
    :param hidden_size:
    :param device:
    :return:
    '''
    return (
        torch.zeros((batch_size, hidden_size), device=device),
        torch.zeros((batch_size, hidden_size), device=device)
    )

def lstm_forward(inputs, state, params):
    '''
    This function is to define the forward function of LSTM
    :param inputs:
    :param state: Contains state and memory
    :param memory:
    :return:
    '''
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hp, b_p = params
    H, C = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_memory = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_memory
        H = O * torch.tanh(C)
        y = H @ W_hp + b_p
        outputs.append(y)
    return torch.cat(outputs, dim=0), (H,C)

class LSTMModelScratch:
    def __init__(self, vocab_size, hidden_size, device, get_params, init_state, forward_fn):
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        self.params = get_params(vocab_size, hidden_size, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, inputs, state):
        inputs = F.one_hot(inputs.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(inputs, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.hidden_size, device)

def lstm_predict(prefix, num_steps, net, vocab, device):
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
    if isinstance(net, nn.Module):
        params = [param for param in net.parameters() if param.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum(param.grad ** 2) for param in params))
    if norm > theta:
        for param in params:
            param.grad *= theta / norm

def train_one_epoch(net, train_loader, loss, optimizer, device, use_random_iter):
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
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        optimizer = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: lstm_predict(prefix, 50, net, vocab, device)
    for epoch in range(epochs):
        perplexity, speed = train_one_epoch(net, train_loader, loss, optimizer, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1}')
            print(predict('timer traveller'))
            print(f'Perplexity: {perplexity}')
            print(f'Speed: {speed} tokens/sec on {str(device)}')

vocab_size, hidden_size, device = len(vocab), 256, torch.device('cuda')
epochs = 500
lr = 1
model_scratch = LSTMModelScratch(vocab_size, hidden_size, device, get_lstm_params, init_lstm_state_memory, lstm_forward)
# train(model_scratch, train_loader, vocab, lr, epochs, device)

'''
Concise Implementation
'''
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
        return (
            torch.zeros((self.num_direction * self.lstm.num_layers, batch_size, self.hidden_size), device=device),
            torch.zeros((self.num_direction * self.lstm.num_layers, batch_size, self.hidden_size), device=device)
        )

num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, hidden_size)
model = LSTMModel(lstm_layer, vocab_size)
model = model.to(device)
train(model, train_loader, vocab, lr, epochs, device)