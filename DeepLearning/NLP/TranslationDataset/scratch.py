import torch
a = torch.arange(1, 10 + 1).reshape((2, -1))
print(a.shape)
a = a.repeat(10, 3)
print(a.shape)

x = torch.tensor([1, 2, 3])
print(x.shape)
print(x.repeat(4, 2, 1).shape)