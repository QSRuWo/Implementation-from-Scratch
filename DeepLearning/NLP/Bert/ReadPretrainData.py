import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from ReadData import tokenize, Vocab
from d2l import torch as d2l

d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    paragraphs = [line.strip().lower().split(' . ') for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

def _get_next_sentence(sentence, next_sentence, paragraphs):
    '''
    To combine a sentence with another sentence.
    :param sentence:
    :param next_sentence:
    :param paragraphs:
    :return:
    '''
    if random.random() < 0.5:
        is_next = True
    else:
        # random.choice: return a random item of a list
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

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

def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    '''
    This function is to combine two sentences
    :param paragraph:
    :param paragraphs:
    :param vocab:
    :param max_len:
    :return:
    '''
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        # There 0.5 possibility that a sentence will be combined with its next sentence, in this case: is_next=True
        # And 0.5 possibility that a sentence randomly combined with another sentence, in this case: is_next=False
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        # segments are used to mark every token its belongings to which sentence
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    '''

    :param tokens:
    :param candidate_pred_positions:
    :param num_mlm_preds: To restrict how many tokens for predicting
    :param vocab:
    :return:
    '''
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # random.random() return a value in [0, 1), which follow uniform distribution
        # With 0.8 probability, set masked_token to '<masked>'
        # With 0.1 probability, set masked_token to its original token
        # With another 0.1 probability, set masked_token to a random other token
        if random.random() < 0.8:
            masked_token = '<masked>'
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                # random.randint(): return a random value in the range of [start, stop]
                masked_token = random.randint(0, len(vocab)-1)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position])
        )
        return mlm_input_tokens, pred_positions_and_labels
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # ?
    # round(): half adjust
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x:x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_labels_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        valid_lens.append(torch.tensor(len(token_ids),dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_labels_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_labels_ids + [0] * (max_num_mlm_preds - len(mlm_pred_labels_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)

class _WikiTextDataste(Dataset):
    def __init__(self, paragraphs, max_len):
        super(_WikiTextDataste, self).__init__()
        paragraphs = [tokenize(paragraph, mode='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next)) for tokens, segments, is_next in examples]
        (self.all_token_ids, self.all_segments, self.valid_lens, self.all_pred_positions,
         self.all_mlm_weight, self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx], self.all_pred_positions[idx],
         self.all_mlm_weight[idx], self.all_mlm_labels[idx], self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

def load_data_wiki(batch_size, max_len):
    data_dir = r'../data/wikitext-2'
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataste(paragraphs, max_len)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    return train_loader, train_set.vocab

if __name__ == '__main__':
    batch_size, max_len = 512, 64
    train_loader, vocab = load_data_wiki(batch_size, max_len)
    for (tokens_X, segments_X, valid_lens_X, pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y) in train_loader:
        print(tokens_X.shape, segments_X.shape, valid_lens_X.shape, pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape, nsp_Y.shape)
        break
    print(len(vocab))