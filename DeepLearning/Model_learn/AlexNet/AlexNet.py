import torch
import torch.nn as nn

# 自己实现
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(11, 11), stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(6400, 4096),
            # 全连接层后面也可以接激活层
            nn.ReLU(),
            # Alexnet在前两个全连接层后面加入了dropout
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 沐神实现
# 训练集为FashionMNIST，所以对通道数做了修改，第三个384的卷积输出从384改为256，第一层全连接输入改为6400，输出层改为10（类）。
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

# X = torch.rand(1, 1, 224, 224)
# for layer in net:
#     X = layer(X)
#     print(f'{layer.__class__.__name__} output shape {X.shape}')

# from torchsummary import summary
device = torch.device('cuda')
def init(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.xavier_uniform_(layer.weight)
net.apply(init)
net.to(device)
# summary(net, (1, 224, 224))

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = FashionMNIST(root='D:/Pytorch_Practice/DeepLearning/CV/Classification/dataset', train=True, download=False, transform=trans)
test_dataset = FashionMNIST(root='D:/Pytorch_Practice/DeepLearning/CV/Classification/dataset', train=False, download=False, transform=trans)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch.optim as optim
from torch.optim import lr_scheduler
lr = 0.01
# optimizer = optim.SGD(net.parameters(), lr=lr)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

epochs = 35
index = 22

train_loss = []
train_acc = []
test_loss = []
test_acc = []
running_info = []

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
    info = f'Epoch {epoch} training loss: {running_loss}, training acc: {running_acc}'
    print(info)
    running_info.append(info)
    # scheduler.step()

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
    info = f'Test loss: {running_loss}, test acc: {running_acc}'
    print(info)
    running_info.append(info)

with open('./fine_tuning_record.txt', 'a') as f:
    f.write(f'Group: {index}; Epoch: {epochs}; Batch Size: {train_loader.batch_size}; lr: {lr}; training loss: {train_loss[-1]}; training acc: {train_acc[-1]}; test loss: {test_loss[-1]}; test acc: {test_acc[-1]};\n')
with open(f'./info/{index}.txt', 'a') as f:
    for info in running_info:
        f.write(info + '\n')


import matplotlib.pyplot as plt

plt.figure()

plt.plot(range(epochs), train_loss, label='Training loss', marker='o', color='red')
plt.plot(range(epochs), train_acc, label='Training acc', marker='o', color='orange')
plt.plot(range(epochs), test_loss, label='Test loss', marker='o', color='purple')
plt.plot(range(epochs), test_acc, label='Test acc', marker='o', color='blue')

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title('Alexnet training and test loss and acc')
plt.show()
