import os
import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from ReadData import tokenize, Vocab, truncate_pad
from d2l import torch as d2l

'''
This python file is to read Stanford Natural Language Inference (SNLI) Corpus
'''

# d2l.DATA_HUB['SNLI'] = (
#     'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
#     '9fcde07509c7e87ec61c640c1b2753d9041758e4')
#
# data_dir = d2l.download_extract('SNLI')

def read_snli(data_dir, is_train):
    '''
    Read SNLI into premises, hypothesis, and labels
    :param data_dir:
    :param is_train:
    :return:
    '''
    def extract_text(s):
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r', encoding='utf-8') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

class SNLIDataset(Dataset):
    def __init__(self, dataset, num_steps, vocab=None):
        super(SNLIDataset, self).__init__()
        self.num_steps = num_steps
        all_premises_tokens = tokenize(dataset[0])
        all_hypothesis_tokens = tokenize(dataset[1])
        # If fine-tuning, vocab here must be consistent with vocab that used in pre-training
        if vocab is None:
            self.vocab = Vocab(all_premises_tokens + all_hypothesis_tokens, min_freq=5, reversed_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premises_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read' + str(len(self.premises)) + 'examples')

    def _pad(self, lines):
        return torch.tensor(
            [truncate_pad(self.vocab[line], self.num_steps, self.vocab['<pad>']) for line in lines]
        )

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)

def load_data_snli(batch_size, num_steps=50):
    train_data = read_snli(data_dir, is_train=True)
    test_data = read_snli(data_dir, is_train=False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, train_set.vocab

if __name__ == '__main__':
    train_data = read_snli(data_dir, is_train=True)
    for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
        print('premise:', x0)
        print('hypothesis:', x1)
        print('label:', y)

    test_data = read_snli(data_dir, is_train=False)
    for data in [train_data, test_data]:
        print([[row for row in data[2]].count(i) for i in range(3)])

    train_loader, test_loader, vocab = load_data_snli(128, 50)

    for X, Y in train_loader:
        print(X[0].shape)
        print(X[1].shape)
        print(Y.shape)
        break