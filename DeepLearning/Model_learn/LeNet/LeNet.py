import torch
import torch.nn as nn
from torchsummary import summary

# 自己实现
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 沐神实现版本
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5,padding=2),
    # nn.Sigmoid(),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    # nn.Sigmoid(),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    # nn.Sigmoid(),
    nn.ReLU(),
    nn.Linear(120, 84),
    # nn.Sigmoid(),
    nn.ReLU(),
    nn.Linear(84, 10)
)

# X = torch.rand((1, 1, 28, 28), dtype=torch.float32)

# # 1. 方法一：如此可以查看每层输出
# for layer in net:
#     X = layer(X)
#     print(f'{layer.__class__.__name__} output shape: {X.shape}')
#
# # 2. 方法二：torchsummary
# net = net.to(torch.device('cuda'))
# summary(net, (1,28,28))

# 初始化权重
def init_weights(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        # nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_normal_(layer.weight)
net.apply(init_weights)

from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

trans = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = FashionMNIST(root='D:\Pytorch_Practice\DeepLearning\CV\Classification\dataset', train=True, transform=trans, download=True)
test_dataset = FashionMNIST(root='D:\Pytorch_Practice\DeepLearning\CV\Classification\dataset', train=False, transform=trans, download=False)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# print(len(train_loader.dataset))
# print(len(test_loader.dataset))

lr = 0.1
optimizer = optim.SGD(net.parameters(), lr=lr)
# optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, verbose=True)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9, verbose=True)
device = torch.device('cpu')
net.to(device)

train_loss = []
train_acc = []
test_loss = []
test_acc = []
running_info = []

epochs = 35
index = 17

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

        _, preds = torch.max(output, 1)
        # 也可以写成
        # preds = output.argmax(axis=1)

        # running_acc += torch.sum(preds == labels.data)
        # 这样之后想要画图，就可以不必把running_acc的计算结果从cpu挪到gpu，节省时间
        cmp = preds == labels.data

        running_acc += float(cmp.sum())

    # scheduler.step()

    running_loss /= len(train_loader)
    # running_acc = running_acc.double() / len(train_dataset)
    running_acc = running_acc / len(train_dataset)
    train_loss.append(running_loss)
    train_acc.append(running_acc)
    info = f'Epoch {epoch} training loss: {running_loss}, training acc: {running_acc}'
    print(info)
    running_info.append(info)

    net.eval()
    running_loss = 0.
    running_acc = 0
    with torch.no_grad():
        for inputs, labels, in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = net(inputs)

            loss = criterion(output, labels)

            running_loss += loss.item()

            _, preds = torch.max(output, 1)

            # running_acc += torch.sum(preds == labels.data)

            cmp = preds == labels.data

            running_acc += float(cmp.sum())

    running_loss /= len(train_loader)
    # 这里犯过一个小错误，分母写错了
    # running_acc = running_acc.double() / len(train_dataset)
    # running_acc = running_acc.double() / len(test_dataset)
    running_acc = running_acc / len(test_dataset)
    test_loss.append(running_loss)
    test_acc.append(running_acc)
    info = f'Test loss: {running_loss}, test acc: {running_acc}'
    print(info)
    running_info.append(info)

with open('./fine_tuning_record.txt','a') as f:
    f.write(f'Group: {index}; Epoch: {epochs}; Batch Size: {train_loader.batch_size}; lr: {lr}; training loss: {train_loss[-1]}; training acc: {train_acc[-1]}; test loss: {test_loss[-1]}; test acc: {test_acc[-1]};\n')
with open(f'./info/{index}.txt', 'a') as f:
    for info in running_info:
        f.write(info+'\n')

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))

# 注意看上方计算acc的改动
# for i, t in enumerate(train_acc):
#     train_acc[i] = t.cpu()
# for i, t in enumerate(test_acc):
#     test_acc[i] = t.cpu()

epochs = range(epochs)

plt.plot(epochs, train_loss, label='Training loss', color='blue', marker='o')
plt.plot(epochs, train_acc, label='Training acc', color='pink', marker='o')
plt.plot(epochs, test_loss, label='Test loss', color='green', marker='o')
plt.plot(epochs, test_acc, label='Test acc', color='yellow', marker='o')

plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training and Test accuracy and loss')
plt.legend()

plt.show()

