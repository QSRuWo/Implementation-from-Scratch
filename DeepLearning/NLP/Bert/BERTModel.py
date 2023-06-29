import torch
from torch import nn
from Model import EncoderBlock

'''
    This is a scaled down version of BERT
'''

def get_tokens_and_segments(tokens_a, tokens_b=None):
    '''
    This is to convert tokens to BERT input form
    :param tokens_a: tokens of sentence 1
    :param tokens_b: tokens of sentence 2 (Optional)
    :return:
    '''
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # segments is to tell the model the belongings (to which sentence) of every token
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class BERTEncoder(nn.Module):
    '''
    In essence, BERT is actually a transformer without decoder
    '''
    def __init__(self, vocab_size, hidden_size, norm_shape, ffn_num_input,
                 ffn_hidden_size, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, hidden_size, norm_shape,
                ffn_num_input, ffn_hidden_size, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      hidden_size))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

class MaskLM(nn.Module):
    '''
    Define the masked language model task of BERT
    '''
    def __init__(self, vocab_size, hidden_size, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, X, pred_positions):
        '''

        :param X:
        :param pred_positions: Record which token to predict. Shape like (batch_size, number of tokens to predict(the idx of tokens in X at dimension 1))
        :return:
        '''
        num_pred_postions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_postions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape(batch_size, num_pred_postions, -1)
        Y_hat = self.mlp(masked_X)
        return Y_hat

class NextSentencePred(nn.Module):
    '''
    Define the next sentence prediction task of BERT. Actually a binary classification task.
    To judge whether two sentence are coherent.
    '''
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)

# class BERTModel(nn.Module):
#     '''
#     Define the complete BERT model
#     '''
#     def __init__(self, vocab_size, hidden_size, norm_shape, ffn_num_input, ffn_hidden_size, num_heads, num_layers, dropout,
#                  max_len=1000, key_size=768, query_size=768, value_size=768, hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
#         super(BERTModel, self).__init__()
#         self.encoder = BERTEncoder(vocab_size, hidden_size, norm_shape, ffn_num_input, ffn_hidden_size, num_heads, num_layers, dropout, max_len=max_len,
#                                    key_size=key_size, query_size=query_size, value_size=value_size)
#         self.hidden = nn.Sequential(nn.Linear(hid_in_features, hidden_size), nn.Tanh())
#         self.mlm = MaskLM(vocab_size, hidden_size, mlm_in_features)
#         self.nsp = NextSentencePred(nsp_in_features)
#
#     def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
#         encoded_X = self.encoder(tokens, segments, valid_lens)
#         if pred_positions is not None:
#             mlm_Y_hat = self.mlm(encoded_X, pred_positions)
#         else:
#             mlm_Y_hat = None
#         nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
#         return encoded_X, mlm_Y_hat, nsp_Y_hat

class BERTModel(nn.Module):
    """The BERT model.

    Defined in :numref:`subsec_nsp`"""
    def __init__(self, vocab_size, hidden_size, norm_shape, ffn_num_input,
                 ffn_hidden_size, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, hidden_size, norm_shape,
                    ffn_num_input, ffn_hidden_size, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, hidden_size),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, hidden_size, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat

if __name__ == '__main__':
    vocab_size, hidden_size, ffn_hidden_size, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2

    encoder = BERTEncoder(vocab_size, hidden_size, norm_shape, ffn_num_input, ffn_hidden_size, num_heads, num_layers, dropout)

    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0,0,0,0,1,1,1,1], [0,0,0,1,1,1,1,1]])
    encoded_X = encoder(tokens, segments, None)
    print(encoded_X.shape)

    mlm = MaskLM(vocab_size, hidden_size)
    mlm_positions = torch.tensor([[1,5,2], [6,1,5]])
    mlm_Y_hat = mlm(encoded_X, mlm_positions)
    print(mlm_Y_hat.shape)

    mlm_Y = torch.tensor([[7,8,9], [10,20,30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1))
    print(mlm_l.shape)

    encoded_X = torch.flatten(encoded_X, start_dim=1)
    nsp = NextSentencePred(encoded_X.shape[-1])
    nsp_Y_hat = nsp(encoded_X)
    print(nsp_Y_hat.shape)

    nsp_y = torch.tensor([0, 1])
    nsp_l = loss(nsp_Y_hat, nsp_y)
    print(nsp_l.shape)