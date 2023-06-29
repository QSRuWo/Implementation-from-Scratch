import torch

# a = torch.arange(10)
# idx = torch.tensor(torch.arange(6)).reshape(2, 3)
# print(a[idx])
# print(a[idx].shape)
a = [1, 2, 3, 4, 5, 6]

def test(*args, **kwargs):
    print(type(args))
    print(args)
    print(*args)
    print(kwargs.keys())

test(*a, b=1)