import torch
from torch import nn
from ReadPretrainData import load_data_wiki
from BERTModel import BERTModel
from tools import Timer, Accumulator
from ReadPretrainData import get_tokens_and_segments

'''
This class is too define a small-scale BERT pre-training
'''

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_X, pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y):
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_X.reshape(-1), pred_positions_X)
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    nsp_l = loss(nsp_Y_hat, nsp_Y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

def train_bert(train_loader, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    step = 0
    timer = Timer()
    metric = Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_X, pred_positions_X, mlm_weight_X, mlm_Y, nsp_Y in train_loader:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_X = valid_lens_X.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weight_X = mlm_weight_X.to(devices[0])
            mlm_Y, nsp_Y = mlm_Y.to(devices[0]), nsp_Y.to(devices[0])
            optimizer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_X, pred_positions_X, mlm_weight_X, mlm_Y, nsp_Y)
            l.backward()
            optimizer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            step += 1

            print(f'Step: {step}')
            print(f'MLM Loss: {metric[0] / metric[3]:.3f}')
            print(f'NSP Loss: {metric[1] / metric[3]:.3f}')
            print(f'Speed: {metric[2] / timer.stop():.1f} sentence paird/sec on {str(devices)}')

            if step == num_steps:
                num_steps_reached = True
                break

def get_bert_encoding(net, device, tokens_a, tokens_b=None):
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=device).unsqueeze(0)
    segments = torch.tensor(segments, device=device).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=device).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X



if __name__ == '__main__':
    batch_size, max_len = 512, 64
    train_loader, vocab = load_data_wiki(batch_size, max_len)

    net = BERTModel(vocab_size=len(vocab), hidden_size=128, norm_shape=[128], ffn_num_input=128, ffn_hidden_size=256,
                    num_heads=2, num_layers=2, dropout=0.2,
                    max_len=1000, key_size=128, query_size=128, value_size=128, hid_in_features=128,
                    mim_in_features=128, nsp_in_features=128)
    devices = [torch.device('cuda')]
    loss = nn.CrossEntropyLoss()

    train_bert(train_loader, net, loss, len(vocab), devices, 50)

    tokens_a, tokens_b = ['a', 'crane', 'is', 'flying'], ['a', 'crane', 'is', 'not', 'flying']
    encoded_text = get_bert_encoding(net, devices[0], tokens_a, tokens_b)
    encoded_text_cls = encoded_text[:, 0, :]
    encoded_text_crane = encoded_text[:, 2, :]
    print(encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0, :3])