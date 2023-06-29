import torch

a = torch.rand((2,2))
print(a[:, None].shape)