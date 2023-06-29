import torch
import torch.utils.data as data
import os
from d2l import torch as d2l
import matplotlib.pyplot as plt
import collections

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_dataset(text):
    '''
    This function is to preprocess the dataset
    :param text:
    :return:
    '''
    # To probe whether there is no space between punctuation and char
    def no_space(char, prev_char):
        return char in (',.!?') and prev_char != ' '
    # text.replace('\u202f', ' ') is replacing all instances of the Unicode character represented by '\u202f' (a narrow non-breaking space) with a regular space.
    # .replace('\xa0', ' ') is then replacing all instances of the Unicode character represented by '\xa0' (a non-breaking space) with a regular space.
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # This is to add regular space between punctuation and char
    # This is for adding punctuations to vocab
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)

def tokenize(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

# Draw the distribution of length of the data
def draw_hist(source, target):
    plt.figure()
    plt.hist([[len(l) for l in source], [len(l) for l in target]], label=['source', 'target'])
    plt.legend()
    plt.show()

class Vocab:
    '''
    Build a map that give each token an index
    '''
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        '''
        :param tokens: A list consists of several lines which composed of several tokens(words or chars)
        :param min_freq: If freq of a token < min_freq, then delete
        :param reserved_tokens:
        '''
        if tokens == None:
            tokens = []
        if reserved_tokens == None:
            reserved_tokens = []
        counter = self.count_freq(tokens)
        # Sort all tokens according to their frequencies
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # Define map between idx and token, token and idx
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token : idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            elif token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        '''
        Give tokens, get their indices
        :param tokens:
        :return:
        '''
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        '''
        Give indices, get the tokens
        :param indices:
        :return:
        '''
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.to_tokens(idx) for idx in indices]

    def count_freq(self, tokens):
        '''
        This function is to count the frequency of every token
        :param tokens:
        :return:
        '''
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def truncate_pad(line, num_steps, padding_token):
    '''
    This function is to truncate or pad the token, to make every token has the same length
    :param line:
    :param num_steps:
    :param padding_token:
    :return:
    '''
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

def build_array(lines, vocab, num_steps):
    '''
    Convert sequence to small batches
    :param lines:
    :param vocab:
    :param num_steps:
    :return:
    '''
    lines = [vocab[line] for line in lines]
    lines = [line + [vocab['<eos>']] for line in lines]
    array = torch.tensor([truncate_pad(line, num_steps, vocab['<pad>']) for line in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_array(data_arrays, batch_size, is_Train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_Train)


def load_data_translate(batch_size, num_steps, num_examples=600):
    text = preprocess_dataset(read_data_nmt())
    source, target = tokenize(text, num_examples)
    src_vocab = Vocab(source, 2, ['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, 2, ['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array(target, tgt_vocab, num_steps)
    data_array = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_loader = load_array(data_array, batch_size)
    return data_loader, src_vocab, tgt_vocab

if __name__ == '__main__':
    train_loader, src_vocab, tgt_vocab = load_data_translate(2, 8)
    for X, X_valid_len, Y, Y_valid_len in train_loader:
        print('X:', X.type(torch.int32))
        print('Valid lengths for X', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Valid lengths for Y', X_valid_len)
        break
