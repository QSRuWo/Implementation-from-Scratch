# a = []
#
# def test():
#     return 1, 2
#
# a.append(test())
# a.append((3,4))
# a.append((5,6))
#
# print(a)
#
# b, c = zip(*a)
# print(b)

import torch
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as Image

# a = np.eye(3)
# a = torch.from_numpy(a)
# print(a.data)
#
# b, c = torch.max(a, 1)
#
# print(b)
# print(c)
#
# a = torch.tensor([0,1,0,1,0])
# b = torch.tensor(([[2, 1], [3,4], [1,2], [4,3],[2,1]]))
# _, c = torch.max(b, 1)
# print(a.shape)
# print(c.shape)
# print(a)
# print(c)
# d = torch.sum(c == a.data)
# print(d)

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import PIL.Image as Image

# img = Image.open("D:\Pytorch_Practice\DeepLearning\CV\Classification\dataset\dogs-vs-cats-small/train\cat\cat.0.jpg")
# img.show()
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])
#
# img = transform(img)
# print(type(img))
#
# img = to_pil_image(img)
# img.show()
# a = torch.tensor([[1.,2.], [3.,4.]], requires_grad=True)
# # print(a)
# # print(a.data)
# # print(a.grad)
# #
# # y = a
# # print(y)
# #
# # y.sum().backward()
# # print(a.grad)
# b = (a > 2).float()
# print(b)

import torch
import torchvision.models as models
import torch.nn as nn

# input = torch.rand(1, 3, 224, 224)
# conv2d = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(0,0))
# output = conv2d(input)
# print(output.shape)
# maxpool = nn.MaxPool2d((2,2))
# output = maxpool(output)
# print(output.shape)



import torch
import torch.nn as nn

# # Define two 2D tensors
# a = torch.tensor([[1, 2], [3, 4]])
# b = torch.tensor([[5, 6], [7, 8]])
#
# # Element-wise multiplication
# print(a * b)
# # tensor([[ 5, 12],
# #         [21, 32]])
#
# # Matrix multiplication
# print(torch.mm(a, b))
# # tensor([[19, 22],
# #         [43, 50]])
#
# a = torch.ones((3,3))
# b = torch.ones((3,))
# print(a+b)
# print(a * b)

# x = torch.arange(4.)
# print(x)
# x.requires_grad = True
# print(x.grad)
#
# # y = 2 * torch.dot(x,x)
# y = x.sum()
# print(y)
# y.backward()
# print(x.grad)

# a = lambda x: x// 1
# print(a * 2)

# def baz(*args, **kwargs):
#     for arg in args:
#         print(arg)
#     for key, value in kwargs.items():
#         print(f"{key} = {value}")
#
# baz(1, 2, 3, a=4, b=5)  # Outputs: 1 2 3 a = 4 b = 5

# x = torch.randn((3,4))
# mask = (torch.randn(x.shape) > 0.5).long()
# print(mask.data)
# print(x[mask])


# def block_1():
#     return nn.Sequential(
#         nn.Linear(10,8),
#         nn.ReLU(),
#         nn.Linear(8,5)
#     )
#
# def block_2():
#     net = nn.Sequential()
#     for i in range(4):
#         net.add_module(f'block_{i}', block_1())
#     net.add_module('Linear', nn.Linear(5, 2))
#     return net
#
# net = block_2()
# for name, param in net.named_parameters():
#     print(name)
#
# print(net.state_dict().keys())
# print(net.state_dict().get('block_0.0.bias'))

# class Base():
#     def __init__(self):
#         self.x = 1
#
#     def F(self):
#         pass
#
# class child(Base):
#     def __init__(self):
#         self.x = 2
#         super(child, self).__init__()
#
#
#     def F_2(self):
#         print(self.x)
#
# c = child()
# c.F_2()

# a = torch.randn((2,2), requires_grad=True).to(torch.device('cuda'))
# print(a)
import numpy as np
# y = [1,2,3,4,5]
# print(2*y)
# print(-1*y)
#
# x = torch.tensor([1,2,3,4,5])
# print(2 * x)
#
# x = np.array(x)
# print(x)
# x = list(x)
# print(x)

# x = np.array([1,2,3])
# print(x.shape)
# y = np.stack([x,x,x], axis=1)
# print(y.shape)
# print(y)

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np
# from PIL import Image
#
# # Start with a 2D grayscale image
# img = np.random.rand(30, 30)
#
# # Apply a colormap (for example, the "hot" colormap)
# img_colored = cm.hot(img)
#
# # img_colored now has shape (30, 30, 4). It is a color image with a fourth channel for transparency (alpha).
# # If you don't want the alpha channel, you can discard it:
#
# img_colored = img_colored[:, :, :3]
#
# # Convert the colors to 8-bit integers:
# img_8bit = (img_colored * 255).astype(np.uint8)
#
# # Convert to a PIL Image
# img_pil = Image.fromarray(img_8bit)
#
# # Show the image
# img_pil.show()

# layer = nn.Conv2d(2,3,(3,3))
#
# # def num_p(layer):
# # # #     return sum(l.numel() for l in layer.parameters() if l.requires_grad==True)
# # # #
# # # # num_p = num_p(layer)
# # # # print(num_p)
#
# x = list([1,2,3])
# x = torch.tensor(x)
# x = x.numpy()
# print(x.numel())

# import torch
# import torch.nn as nn
# from torchsummary import summary
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.line = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.Linear(64 * 32 * 32, 10)  # Assuming input size is 32*32
#         )
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64*32*32, 10)  # Assuming input size is 32*32
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = x.view(x.size(0), -1)  # Flatten layer
#         x = self.fc1(x)
#         return self.line(x)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # if you have GPU
# model = Net().to(device)
#
# # summary(model, (3, 32, 32))
# print((next(model.parameters()).device))
# print(model.conv1.weight.device)
# model = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.Linear(64 * 32 * 32, 10)  # Assuming input size is 32*32
#         )
# print(model[0])
# a = list(range(10))
# # b = a+1
# # print(b)

# a = list([torch.rand(1), torch.rand(1), torch.rand(1)])
# device = torch.device('cuda')
# for i, t in enumerate(a):
#     a[i] = t.to(device)
# print(a[0].device)
#
# for i, t in enumerate(a):
#     a[i] = t.cpu()
#
# print(a[0].device)
#
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(range(3), a)
#
# plt.show()
# a = torch.tensor([1,0,1])
# b = torch.tensor([1,1,1])
# c = a==b
# print(c)
# d = float(c.sum())
# print(d)
# print(a+1)
# a = a.numpy()
# print(a+1)

# X = torch.rand(1,96,54,54)
# layer = nn.MaxPool2d(kernel_size=3, stride=2)
# print(layer(X).shape)
from torch.nn import functional as F

# class Inception(nn.Module):
#     def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
#         super(Inception, self).__init__()
#         self.p_1 = nn.Conv2d(in_channels, c1, kernel_size=(1,1))
#         self.p_2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=(1,1))
#         self.p_2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=(3, 3), padding=1)
#         self.p_3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=(1, 1))
#         self.p_3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=(5, 5), padding=2)
#         self.p_4_1 = nn.MaxPool2d(kernel_size=(3, 3),stride=1, padding=1)
#         self.p_4_2 = nn.Conv2d(in_channels, c4, kernel_size=(1, 1))
#
#     def forward(self, x):
#         p1 = F.relu(self.p_1(x))
#         p2 = F.relu(self.p_2_2(F.relu(self.p_2_1(x))))
#         p3 = F.relu(self.p_3_2(F.relu(self.p_3_1(x))))
#         p4 = F.relu(self.p_4_2(self.p_4_1(x)))
#
#         return torch.cat((p1, p2, p3, p4), dim=1)
#
# # net = nn.Sequential(Inception(in_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32))
# net = Inception(in_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32)
#
# for layer in net:
#     print(f'{layer.__class__.__name__}')

# layer = nn.Conv2d(1,1,kernel_size=(7,7), stride=2, padding=3)
layer = nn.MaxPool2d(2)
x = torch.rand((1,1,224,224))
y = layer(x)
print(y.shape)
print(layer.kernel_size)