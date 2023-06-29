import torch
import torch.utils.data as data
import collections

def read_data(root=r'../data/fra-eng/fra.txt'):
    '''
    Read the file char by char to get raw_text
    :param root:
    :return:
    '''
    with open(root, 'r', encoding='utf-8') as f:
        return f.read()

def process_raw_text(raw_text):
    '''
    This dataset is to standardize the form the raw text and
    add space between char and punctuations
    :param text:
    :return:
    '''
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '
    # Standardize the space and make every letter in lower form
    text = raw_text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Add space between char and punctuations
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)

def tokenize(text, num_samples=None):
    '''
    This function is to tokenize the text at the unit of word
    :param text:
    :param num_samples: Maximum tokens
    :return:
    '''
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_samples and i > num_samples:
            break
        part = line.split('\t')
        if len(part) == 2:
            source.append(part[0].split(' '))
            target.append(part[1].split(' '))
    return source, target

class Vocab:
    '''
    Build a map between tokens and their unique index
    '''
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        '''
        :param tokens:
        :param min_freq: if frequency of a token < min_freq, then delete it
        :param reserved_tokens:
        '''
        if tokens == None:
            tokens = []
        if reserved_tokens == None:
            reserved_tokens = []
        # Calculate frequency of every token and sort the result list
        count = self.count_freq(tokens)
        self._tokens_freq = sorted(count.items(), key=lambda x: x[1], reverse=True)
        # Build the map between tokens and indices
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token : idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._tokens_freq:
            if freq < min_freq:
                break
            elif token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def count_freq(self, tokens):
        '''
        This function is to count frequency of every token
        :param tokens:
        :return:
        '''
        if len(tokens) == 0 or isinstance(tokens, list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_token(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.to_token(idx) for idx in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._tokens_freq

def truncate_pad(line_of_token, num_steps, padding_token):
    '''
    This function is to make sure every line of tokens has the same length
    :param line_of_token:
    :param num_steps:
    :param padding_token:
    :return:
    '''
    if len(line_of_token) > num_steps:
        return line_of_token[:num_steps]
    return line_of_token + [padding_token] * (num_steps - len(line_of_token))

def build_array(lines_of_tokens, vocab, num_steps):
    '''
    Convert the tokens sequences to small batches
    :param lines_of_tokens:
    :param vocab:
    :param num_steps:
    :return:
    '''
    # Convert tokens to their indices
    lines_of_indices = [vocab[line_of_token] for line_of_token in lines_of_tokens]
    # Add vocab['<eos>'] as ending of one 'sentence' to every line
    lines_of_indices = [line_of_indices + [vocab['<eos>']] for line_of_indices in lines_of_indices]
    # Create tensor
    array = torch.tensor([truncate_pad(line_of_indices, num_steps, vocab['<pad>']) for line_of_indices in lines_of_indices])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(dim=1)
    return array, valid_len

def load_array(array, batch_size, is_Train=True):
    '''
    Convert data tensor to dataloader
    :param array:
    :param batch_size:
    :param is_Train:
    :return:
    '''
    dataset = data.TensorDataset(*array)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_Train)

def load_data_translate(batch_size, num_steps, num_examples=600):
    text = process_raw_text(read_data())
    source, target = tokenize(text, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
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
        print('Y:', Y.shape)
        print('Valid lengths for Y', Y_valid_len)
        break
