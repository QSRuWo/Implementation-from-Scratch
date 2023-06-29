import torch.nn as nn

class Encoder(nn.Module):
    '''
    Define base encoder interface for encoder-decoder architecture
    '''
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    '''
    Define base decoder interface for encoder-decoder architecture
    '''
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, encoder_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise  NotImplementedError