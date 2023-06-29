import torch
import torch.nn as nn

# 自己实现
def vgg_block(in_channels, out_channels, num_conv):
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

channels = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]

def vgg16(channels):
    layers = []
    in_channels = 1
    for num_conv, out_channels in channels:
        layers.append(vgg_block(in_channels, out_channels, num_conv))
        in_channels = out_channels

    return nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10),
    )

# 沐神版本
def vgg_block_mu(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block_mu(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 10))

# 沐神版本将通道数缩小了4倍因为VGG11的网络比Alexnet大，节约时间
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]

net = vgg(small_conv_arch)

# X = torch.rand(1, 1, 224, 224)
# for layer in net:
#     X = layer(X)
#     print(f'{layer.__class__.__name__} output shape {X.shape}')

from torchsummary import summary
device = torch.device('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def init(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.xavier_uniform_(layer.weight)
net.apply(init)
net.to(device)

summary(net, (1, 224, 224))

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

optimizer = optim.SGD(net.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

epochs = 1

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

# import matplotlib.pyplot as plt
#
# plt.figure()
#
# plt.plot(range(epochs), train_loss, label='Training loss', marker='o', color='red')
# plt.plot(range(epochs), train_acc, label='Training acc', marker='o', color='orange')
# plt.plot(range(epochs), test_loss, label='Test loss', marker='o', color='purple')
# plt.plot(range(epochs), test_acc, label='Test acc', marker='o', color='blue')
#
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Value")
# plt.title('Vgg11 training and test loss and acc')
# plt.show()