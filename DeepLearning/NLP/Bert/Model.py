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

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, hidden_size))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, hidden_size, 2, dtype=torch.float32) / hidden_size)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, hidden_size, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, hidden_size, bias=bias)
        self.W_k = nn.Linear(key_size, hidden_size, bias=bias)
        self.W_v = nn.Linear(value_size, hidden_size, bias=bias)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, values is like:
        # (batch_size, number of q/k/v, hidden_size)
        # Shape of valid_lens is like:
        # (batch_size,) or (batch_size, number_of_queries)
        # After transpose, q/k/v shape is like:
        # (batch_size * num_heads, number_of_q/k/v, hidden_size / num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)
        # The output shape is like:
        # (batch_size * num_heads, number_of_queries, hidden_size / num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    '''
    Reverse the operation of transpose_qkv
    :param X:
    :param num_heads:
    :return:
    '''
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class AddNorm(nn.Module):
    '''
    Do Residual add and Normalization
    '''
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_hidden_size, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_hidden_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_hidden_size, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class EncoderBlock(nn.Module):
    '''
    Define Encoder
    '''
    def __init__(self, key_size, query_size, value_size, hidden_size,
                 norm_shape, ffn_num_input, ffn_hidden_size, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, hidden_size, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_hidden_size, hidden_size)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class DecoderBlock(nn.Module):
    '''
    Define Decoder
    '''
    def __init__(self, key_size, query_size, value_size, hidden_size, norm_shape, ffn_num_input, ffn_hidden_size, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention_1 = MultiHeadAttention(key_size, query_size, value_size, hidden_size, num_heads, dropout)
        self.add_norm_1 = AddNorm(norm_shape, dropout)
        self.attention_2 = MultiHeadAttention(key_size, query_size, value_size, hidden_size, num_heads, dropout)
        self.add_norm_2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_hidden_size, hidden_size)
        self.add_norm_3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            decoder_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            decoder_valid_lens = None
        X2 = self.attention_1(X, key_values, key_values, decoder_valid_lens)
        Y = self.add_norm_1(X, X2)
        Y2 = self.attention_2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.add_norm_2(Y, Y2)
        return self.add_norm_3(Z, self.ffn(Z)), state

class EncoderDecoder(nn.Module):
    '''
    Define the complete encoder decoder architecture
    '''
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input, *args):
        encoder_output = self.encoder(encoder_input, *args)
        decoder_state = self.decoder.init_state(encoder_output, *args)
        return self.decoder(decoder_input, decoder_state)