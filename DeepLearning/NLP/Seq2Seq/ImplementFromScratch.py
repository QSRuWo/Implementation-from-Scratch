import torch
import torch.nn as nn
from Interface import Encoder, Decoder
from tools import Timer, Accumulator, grad_clipping
from ReadData import load_data_translate, truncate_pad
import math
import collections
from d2l import torch as d2l

class Seq2SeqEncoder(Encoder):
    '''
    Define Sequence to Sequence Encoder architecture.
    '''
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # There are vocab_size kinds of embedding(One vocab idx of token corresponds to one embedding),
        # and each embedding has a size of embed_size(For example, 3, then a embedding is like [1, 2, 3], this is just a example,
        # and this [1, 2, 3] is like a ID Card of this vocab idx, or this token.)
        # The kinds of values in input should not larger than vocab_size, or there will be an IndexError
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, X, *args):
        # You can superficially count embedding layer as a kind of one-hot, but this layer can learn.
        X = self.embedding(X)
        # Move sequence length to the first dimension and batch size in the second dimension
        X = X.permute(1, 0, 2)
        # Pytorch will default an initial state of zero. (RNN, GRU, LSTM)
        output, state = self.rnn(X)
        return output, state

class Seq2SeqDecoder(Decoder):
    '''
    Define Sequence to Sequence Decoder architecture
    '''
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # Decode will combine the state output by encoder and input into Decode
        self.rnn = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers, dropout=dropout)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.v = vocab_size

    def init_state(self, encoder_outputs, *args):
        # Encoder_outputs is composed by (output, state)
        # So here we acquire state
        return encoder_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        # State stores all the states output by every layer in encoder's rnn,
        # and here state[-1] means state of the last layer
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_context, state)
        # nn.Linear not only accept 2-dimension input
        output = self.dense(output).permute(1, 0, 2)
        return output, state

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
    lr, num_epochs, device = 0.005, 10, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = load_data_translate(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                             dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                             dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    train_Seq2Seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_Seq2Seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')