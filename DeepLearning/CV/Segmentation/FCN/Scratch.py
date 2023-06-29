import torch
import torch.nn as nn
import torch.nn.functional as F

# inputs = torch.randn((16, 21, 320, 480)) # generates a tensor with values from a normal distribution
# targets = torch.randint(0, 10, (16, 320, 480)) # generates a tensor with random integers between 0 and 20
#
# # loss = F.cross_entropy(inputs, targets,reduction='none').mean(1).mean(1).sum()
# # print(loss)
# #
# # criterion = nn.CrossEntropyLoss(reduction='none')
# #
# # loss = criterion(inputs, targets).mean(1).mean(1).sum()
# # print(loss)
# idx = torch.argmax(inputs, dim=1)
# print(idx.shape)
# a = torch.arange(10).reshape(-1, 2, 5)
# a[0, 0, 1] = 11
# b = torch.argmax(a, dim=0)
# print(a)
# print(b)
# print(b.shape)
a = torch.randint(0, 10, (16, 320, 480))
b = torch.randn((10, 3))
c = b[a, :]
print(c.shape)
