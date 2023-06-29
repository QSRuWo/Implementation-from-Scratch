import re
import random
import collections
from d2l import torch as d2l
import torch

'''
    Summary of how to pre-process sequence data:
    1. Read lines from .txt file
    2. Tokenize these lines, at the unit of 'word' or 'char', etc.
    3. Build vocabulary of all tokens, which record the map between all kinds of tokens to their unique indices
    4. Convert all tokens to their indices, and this set called corpus.
    5. Use corpus to build a dataloader
    6. Return this dataloader and its vocab.
'''

def read_time_machine(root):
    '''
    This function is to read timemachine.txt
    :param root:
    :return:
    '''
    with open(root, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, mode='word'):
    '''
    This function is to convert lines of text to tokens
    :param lines:
    :param mode:
    :return:
    '''
    if mode == 'word':
        return [line.split() for line in lines]
    elif mode == 'char':
        # space will be included
        return [list(line) for line in lines]
    else:
        print('Unknown token mode', str(mode))

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

def load_corpus_time_machine(max_tokens):
    '''
    This function is to return all tokens' indices and vocabulary
    :param max_tokens: If >0, to control the size of dataset
    :return:
    '''
    lines = read_time_machine(r'D:\Pytorch_Practice\DeepLearning\NLP\data\timemachine.txt')
    tokens = tokenize(lines, mode='char')
    vocab = Vocab(tokens)
    # Convert all tokens to their indices
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    '''
    This function is to iter corpus randomly
    :param corpus:
    :param batch_size:
    :param num_steps:
    :return:
    '''
    # Start with a random offset, inclusive of num_steps - 1, to partition the sequence
    corpus = corpus[random.randint(0, num_steps):]
    # Subtract one to avoid out of boundary, since the last 'group' have no 'one behind last group' to be its label
    num_seqs = (len(corpus) - 1) // num_steps
    # Get all the starting indices
    initial_indices = list(range(0, num_seqs * num_steps, num_steps))
    # To make these indices random
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos : pos + num_steps]

    num_batches = num_seqs // batch_size
    # Tips: By default, batch size here is smaller than num_seqs
    for i in range(0, num_batches * batch_size, batch_size):
        initial_indices_per_batch = initial_indices[i : i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    '''
    This function is to iter corpus sequentially
    :param corpus:
    :param batch_size:
    :param num_steps:
    :return:
    '''
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset : offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1 : offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i : i + num_steps]
        Y = Ys[:, i : i + num_steps]
        yield X, Y

class SeqDataLoader:
    '''
    Use corpus and vocab to build a data loader
    '''
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        # Choose which iter to be used
        if use_random_iter:
            self.data_iter = seq_data_iter_random
        else:
            self.data_iter = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_randon_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_randon_iter, max_tokens)
    return data_iter, data_iter.vocab

if __name__ == '__main__':
    line_1 = read_time_machine(r'D:\Pytorch_Practice\DeepLearning\NLP\data\timemachine.txt')
    token_1 = tokenize(line_1, mode='char')
    vocab_1 = Vocab(token_1)
    vocab_2 = d2l.Vocab(token_1)
    print(vocab_1.token_freqs == vocab_2.token_freqs)
    c1,v1 = load_corpus_time_machine(0)
    c2,v2 = d2l.load_corpus_time_machine(0)
    print(c1==c2)
    load_1, v_1 = load_data_time_machine(2, 5)
    load_2, v_2 = d2l.load_data_time_machine(2, 5)
    # print(v_1['a'] == v_2.token_freqs['a'])
    print(v_1['b'])
    print(v_2['b'])