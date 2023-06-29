import torch
# a = torch.arange(10).reshape((2,5))
# b = torch.cat((a, a), dim=1)
# print(b.shape)
# print(b)

import torch.nn as nn
# x = torch.randn(16, 3)  # raw scores for two classes for 16 instances
# y = torch.randint(0, 2, (16,))  # class labels (0 or 1) for 16 instances
# loss = nn.CrossEntropyLoss(reduction='none')
# z = loss(x, y)
# print(z.shape)
# z = z.reshape(z.shape[0], -1)
# print(z.shape)
# a = z.mean(dim=1)
# print(a.shape)

# b = a.argmax(dim=1)
# print(b.dtype)
# print(b.type(torch.float32).dtype)
# a = torch.tensor([[1, 2, 3], [3, 4, 5]])
# print(a.shape)
# b = torch.tensor([[1, 2, 3]])
# print(b.shape)
# c = a==b
# print(c)
# a = torch.tensor([[2, 3], [5, 4]])
# b = torch.tensor([[1,5,6,9], [10, 11, 12, 13], [14,15,16,17]])
# # print(a.shape)
# # print(b.shape)
# # c = torch.max(a[:, None, :], b[:, :])
# # # d = torch.max(a[:, :], b[:, :])
# # print(c.shape)
# a = torch.arange(8).reshape((8,1))
#
# print(a)
# # print(a)
# # values, indices = torch.max(a, dim=1)
# # print(values)
# # print(indices)
#
# max_v, indices = torch.max(a, dim=1)
# print(max_v.shape)
# print(torch.nonzero(max_v>=5).reshape(-1))
# print(indices)
# print(max_v>=5)
# print(indices[max_v>=5])
# print(d.shape)
# print(a[:,0] * b[:, 0])
# a = torch.tensor(torch.arange(96)).reshape(2,3,4,4)
# print(a)
# b = a.permute((0, 2, 3, 1))
# print(a.shape)
# print(b.shape)
# print(torch.flatten(a, start_dim=1))
# print(torch.flatten(b, start_dim=1))
import torchvision
val_data = torchvision.io.read_image(r'D:\Pytorch_Practice\DeepLearning\CV\ObjectDetection\data\banana-detection\bananas_val\images\0.png').unsqueeze(0)
print(val_data.shape)
import matplotlib.pyplot as plt
# plt.imshow(val_data.squeeze(0).permute(1, 2, 0))
# plt.show()
import PIL.Image as Image
import torchvision.transforms.functional as F
img = Image.open(r'D:\Pytorch_Practice\DeepLearning\CV\ObjectDetection\data\banana-detection\bananas_val\images\0.png')
img = F.to_tensor(img)
print(img.shape)
import cv2
img = cv2.imread(r'D:\Pytorch_Practice\DeepLearning\CV\ObjectDetection\data\banana-detection\bananas_val\images\0.png')
print(img.shape)
print(type(img))
a = torch.tensor((2,3))
print(a)