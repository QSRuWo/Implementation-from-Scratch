import math
import torch
from torch import nn
import pandas as pd
from Interface import Encoder, AttentionDecoder
from Model import PositionalEncoding, EncoderBlock, DecoderBlock, EncoderDecoder
from ReadData import load_data_translate, truncate_pad
from tools import Timer, Accumulator, grad_clipping
import collections

class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, hidden_size, norm_shape,
                 ffn_num_input, ffn_hidden_size, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                'block' + str(i),
                EncoderBlock(key_size, query_size, value_size, hidden_size, norm_shape, ffn_num_input, ffn_hidden_size, num_heads, dropout, use_bias)
            )

    def forward(self, X, valid_lens, *args):
        X = self.position_encoding(self.embedding(X) * math.sqrt(self.hidden_size))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, hidden_size, norm_shape, ffn_num_input, ffn_hidden_size, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                'block' + str(i),
                DecoderBlock(key_size, query_size, value_size, hidden_size, norm_shape, ffn_num_input, ffn_hidden_size,
                             num_heads, dropout, i)
            )
        self.dense = nn.Linear(hidden_size, vocab_size)

    def init_state(self, encoder_outputs, encoder_valid_lens, *args):
        return [encoder_outputs, encoder_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.position_encoding(self.embedding(X) * math.sqrt(self.hidden_size))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention_1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention_2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

def sequence_mask(X, valid_len, value=0):
    '''
    This function is to process those not valid values
    :param X:
    :param valid_len:
    :param value:
    :return:
    '''
    max_len = X.shape[1]
    mask = torch.arange(max_len, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    '''
    This class is to multiply weights to loss, to delete the loss of padding values.
    '''
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def train_Seq2Seq(net, train_loader, lr, epochs, tgt_vocab, device):
    '''
    Define the training process of sequence to sequence model
    :param net:
    :param train_loader:
    :param lr:
    :param epochs:
    :param tgt_vocab:
    :param device:
    :return:
    '''
    def xavier_init_weight(layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
        if type(layer) == nn.GRU:
            for param in layer._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(layer._parameters[param])

    net.apply(xavier_init_weight)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(epochs):
        timer = Timer()
        metric = Accumulator(2)
        for batch in train_loader:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            # remove the last element in Y, and insert bos in the front of Y
            decoder_input = torch.cat((bos, Y[:, :-1]), dim=1)
            # X_valid_len is not useful here. It is for later implementation of attention
            Y_hat, _ = net(X, decoder_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1}\n'
                  f'Loss: {metric[0] / metric[1]:.3f}\n'
                  f'Speed: {metric[1] / timer.stop()} tokens/sec on {str(device)}')

def predict_Seq2Seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=None):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    encoder_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    encoder_inputs = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # Here we do not directly send all inputs into the whole net, since we should define inputs for decoder in the process of prediction
    encoder_outputs = net.encoder(encoder_inputs, encoder_valid_len)
    decoder_state = net.decoder.init_state(encoder_outputs, encoder_valid_len)
    decoder_inputs = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_sequence, attention_weight_sequence = [], []
    for _ in range(num_steps):
        Y, decoder_state = net.decoder(decoder_inputs, decoder_state)
        decoder_inputs = Y.argmax(dim=2)
        pred = decoder_inputs.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_sequence.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_sequence.append(pred)
    return ' '.join(tgt_vocab.to_token(output_sequence)), attention_weight_sequence

def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i : i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i : i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i : i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

if __name__ == '__main__':
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, 'cuda'
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = load_data_translate(batch_size, num_steps)

    encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size,
                                 num_hiddens, norm_shape, ffn_num_input,
                                 ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size,
                                 num_hiddens, norm_shape, ffn_num_input,
                                 ffn_num_hiddens, num_heads, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_Seq2Seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_Seq2Seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {bleu(translation, fra, k=2):.3f}')