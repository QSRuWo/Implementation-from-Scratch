import torch
import collections
from d2l import torch as d2l
# a = torch.arange(10).reshape((-1, 5))
# print(a)
# b = a.reshape((5,2))
# print(b)

a = torch.rand((2,3,4))
print(torch.sum(a ** 2) == torch.sum((a ** 2)))
model = d2l.RNNModel(gru_layer, len(vocab))