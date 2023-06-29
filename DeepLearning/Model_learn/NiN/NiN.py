import torch
import torch.nn as nn

# 自己实现
# def NiN_block(in_channels, out_channels, kernel_size, stride, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
#         nn.ReLU(),
#         nn.Conv2d(out_channels, out_channels, kernel_size=1),
#         nn.ReLU(),
#         nn.Conv2d(out_channels, out_channels, kernel_size=1),
#         nn.ReLU(),
#     )
#
# conv_arch = [(96, 11, 4, 0), (256, 5, 1, 1), (384, 3, 1, 1), (10, 3, 1, 1)]
#
# def build_NiN(conv_arch):
#     nin_block = []
#     in_channels = 1
#     for out_channels, kernel_size, stride, padding in conv_arch:
#         nin_block.append(NiN_block(in_channels, out_channels, kernel_size, stride, padding))
#         # 最后一个NiN区块没有池化，池化不要写在for循环里，看沐神版本改正
#         nin_block.append(nn.MaxPool2d(kernel_size=3, stride=2))
#         in_channels = out_channels
#
#     return nn.Sequential(
#         *nin_block,
#         nn.AdaptiveAvgPool2d((1,1))
#         # 最后没有flatten，这样结果就会是四维向量，没法计算交叉熵损失
#     )

# model = build_NiN(conv_arch)
#
# X = torch.rand((1,1,224,224))
# for layer in model:
#     X = layer(X)
#     print(f'{layer.__class__.__name__} outputshape {X.shape}')

# 沐神版本
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU())

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())


X = torch.rand((1,1,224,224))
for layer in net:
    X = layer(X)
    print(f'{layer.__class__.__name__} outputshape {X.shape}')

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

train_dataset = FashionMNIST(root='D:/Pytorch_Practice/DeepLearning/CV/Classification/dataset', train=True, download=False, transform=trans)
test_dataset = FashionMNIST(root='D:/Pytorch_Practice/DeepLearning/CV/Classification/dataset', train=False, download=False, transform=trans)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

import torch.optim as optim

# 学习率原来设成0.5不收敛
optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

epochs = 5

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
plt.title('NiN training and test loss and acc')
plt.show()