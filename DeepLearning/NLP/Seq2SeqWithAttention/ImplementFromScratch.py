import torch
from torch import nn
from Interface import Encoder, AttentionDecoder
from Model import AdditiveAttention
from tools import Timer, Accumulator, grad_clipping
from ReadData import load_data_translate, truncate_pad
import math
import collections
from d2l import torch as d2l

class Seq2SeqEncoder(Encoder):
    '''
    Define Seq2Seq Encoder model
    '''
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X).permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state

class Seq2SeqAttentionDecoder(AttentionDecoder):
    '''
    Define Seq2Seq Decoder Model with Attention
    '''
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(hidden_size, hidden_size, hidden_size, dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers, dropout=dropout)
        self.dense = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_output, enc_valid_lens, *args):
        outputs, hidden_state = enc_output
        # return shape (batch_size, num_seqs, hidden_size of encoder.rnn)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        # X shape after permute (num_seqs, batch_size, hidden_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

class EncoderDecoder(nn.Module):
    '''
    Define Complete encoder-decoder model
    '''
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input, *args):
        encoder_output = self.encoder(encoder_input)
        decoder_state = self.decoder.init_state(encoder_output, *args)
        return self.decoder(decoder_input, decoder_state)

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
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = load_data_translate(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens,
                                 num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens,
                                      num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_Seq2Seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_Seq2Seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
    attention_weights = torch.cat(
        [step[0][0][0] for step in dec_attention_weight_seq], 0).reshape(
        (1, 1, -1, num_steps))
    d2l.show_heatmaps(
        attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
        xlabel='Key posistions', ylabel='Query posistions')