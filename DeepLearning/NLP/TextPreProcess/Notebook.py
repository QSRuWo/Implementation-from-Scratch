import collections
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine():
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'Length of lines: {len(lines)}')
print(lines[0])
print(lines[10])

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split(' ') for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('Unknown class: ' + token)

tokens= tokenize(lines)
for i in range(11):
    print(tokens[i])

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens == None:
            tokens = []
        if reserved_tokens == None:
            reserved_tokens = []
        counter = self.count_freq(tokens)
        self.tokens_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.tokens_freq if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[idx] for idx in indices]

    def count_freq(self, tokens):
        '''
        Count frequency of every token
        :param tokens:
        :return:
        '''
        if len(tokens) == 0 or isinstance(tokens[0], list):
            token = [token for line in tokens for token in line]
        return collections.Counter(token)

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

# Convert tokens to idx
for i in [0, 10]:
    print(f'word: {tokens[i]}')
    print(f'indices: {vocab[tokens[i]]}')

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的标记索引列表和词汇表。"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print('corpus', len(corpus))
print('vocab', len(vocab))

