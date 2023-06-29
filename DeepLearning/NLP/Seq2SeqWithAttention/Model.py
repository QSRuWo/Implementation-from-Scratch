import torch
from torch import nn
import torch.nn.functional as F
import math

def sequence_mask(X, valid_lens, value=0):
    '''
    Set all padding value to 'value'
    :param X: A two dimension tensor
    :param valid_len:
    :param value:
    :return:
    '''
    max_len = X.shape[1]
    mask = torch.arange(max_len, dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    '''
    Do softmax with eliminating padding values
    :param X:
    :param valid_len:
    :return:
    '''
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return F.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    '''
    Define Additive Attention model
    '''
    def __init__(self, key_size, query_size, hidden_size, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.w_k = nn.Linear(key_size, hidden_size, bias=False)
        self.w_q = nn.Linear(query_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # Uniform length
        # After linear layer, queries shape is (batch_size, num_queries, hidden_size)
        # keys shape is (batch_size, num_keys, hidden_size)
        queries, keys = self.w_q(queries), self.w_k(keys)
        # queries (batch_size, num_queries, 1, hidden_size)
        # keys (batch_size, 1, num_keys, hidden_size) for broadcast
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # scores (batch_size, num_queries, num_keys)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    '''
    Define Dot Product Attention model
    This class assume queries and keys has the same length
    queries (batch_size, num_queries, length)
    keys (batch_size, num_keys, length)
    '''
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)