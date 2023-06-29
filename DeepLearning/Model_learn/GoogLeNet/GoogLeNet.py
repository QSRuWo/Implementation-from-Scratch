import torch
import torch.nn as nn
from torch.nn import functional as F

# 自己实现
# 这次复现问题主要在于padding不知道是多少，以及调用函数的名字写错了，见下方全局池化
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
# class GoogLeNet(nn.Module):
#     def __init__(self):
#         super(GoogLeNet, self).__init__()
#         self.stage_1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=(7, 7), stride=2, padding=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
#             nn.Conv2d(64, 64, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 192, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
#         )
#         self.stage_2 = nn.Sequential(Inception(in_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32))
#         self.stage_3 = nn.Sequential(Inception(in_channels=256, c1=128, c2=(128, 192), c3=(32, 96), c4=64))
#
#         self.maxpool_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
#
#         self.stage_4 = nn.Sequential(
#             Inception(in_channels=480, c1=192, c2=(96, 208), c3=(16, 48), c4=64),
#             Inception(in_channels=512, c1=160, c2=(112, 224), c3=(24, 64), c4=64),
#             Inception(in_channels=512, c1=128, c2=(128, 256), c3=(24, 64), c4=64),
#             Inception(in_channels=512, c1=112, c2=(144, 288), c3=(32, 64), c4=64),
#             Inception(in_channels=528, c1=256, c2=(160, 320), c3=(32, 128), c4=128)
#         )
#
#         self.maxpool_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
#
#         self.stage_5 = nn.Sequential(
#             Inception(in_channels=832, c1=256, c2=(160, 320), c3=(32, 128), c4=128),
#             Inception(in_channels=832, c1=384, c2=(192, 384), c3=(48, 128), c4=128)
#         )
#         # 这里原来写成了
#         # self.avgpool_1 = nn.AvgPool2d((1, 1))
#         self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.dropout_1 = nn.Dropout(p=0.4)
#
#         self.flatten = nn.Flatten()
#
#         self.linear = nn.Linear(1024, 10)
#
#         self.net = nn.Sequential(
#             self.stage_1,
#             self.stage_2,
#             self.stage_3,
#             self.maxpool_1,
#             self.stage_4,
#             self.maxpool_2,
#             self.stage_5,
#             self.avgpool_1,
#             self.dropout_1,
#             self.flatten,
#             self.linear
#         )
#
#     def forward(self, x):
#         x = self.net(x)
#         return x

# net = GoogLeNet().net
#
# X = torch.rand((1,1,96,96))
# for layer in net:
#     X = layer(X)
#     print(f'{layer.__class__.__name__} outputshape {X.shape}')

# 沐神版本
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2,
                                           padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

device = torch.device('cuda')
# !!!重点，NiN和GoogLeNet这里如果采用torch默认初始化无法收敛，需要用xavier初始化
def init(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.xavier_uniform_(layer.weight)
net.apply(init)
net.to(device)

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = FashionMNIST(root='D:\Pytorch_Practice\DeepLearning\Model_learn\data', train=True, download=False, transform=trans)
test_dataset = FashionMNIST(root='D:\Pytorch_Practice\DeepLearning\Model_learn\data', train=False, download=False, transform=trans)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

epochs = 10

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    net.train()
    running_loss = 0.
    running_acc = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        output = net(inputs)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = output.argmax(axis=1)

        cmp = preds == labels.data

        running_acc += float(cmp.sum())

    running_loss /= len(train_loader)
    running_acc /= len(train_loader.dataset)
    train_loss.append(running_loss)
    train_acc.append(running_acc)
    print(f'Epoch {epoch} training loss: {running_loss}, training acc: {running_acc}')

    net.eval()
    running_loss = 0.
    running_acc = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = net(inputs)

            loss = criterion(output, labels)

            running_loss += loss.item()

            preds = output.argmax(axis=1)

            cmp = preds == labels.data

            running_acc += float(cmp.sum())

    running_loss /= len(train_loader)
    running_acc = running_acc / len(test_dataset)
    test_loss.append(running_loss)
    test_acc.append(running_acc)
    print(f'Test loss: {running_loss}, test acc: {running_acc}')

import matplotlib.pyplot as plt

plt.figure()

plt.plot(range(epochs), train_loss, label='Training loss', marker='o', color='red')
plt.plot(range(epochs), train_acc, label='Training acc', marker='o', color='orange')
plt.plot(range(epochs), test_loss, label='Test loss', marker='o', color='purple')
plt.plot(range(epochs), test_acc, label='Test acc', marker='o', color='blue')

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title('GoogLeNet training and test loss and acc')
plt.show()