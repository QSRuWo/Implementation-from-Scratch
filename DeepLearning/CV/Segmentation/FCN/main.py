import torch
import torch.nn.functional as F
import torch.optim as optim
from Read_Data import create_voc_loader, VOC_COLORMAP
from tools import Accumulator, Timer
from d2l import torch as d2l
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from Model import build_net

# Step_1: Build data loader
batch_size = 16
crop_size = (320, 480)
train_loader, val_loader = create_voc_loader(batch_size, crop_size)
# train_loader, test_iter = d2l.load_data_voc(batch_size, crop_size)

# Step_2: Build net
net = build_net()

def criterion(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-3)
epochs = 5
device = torch.device('cuda')
net = net.to(device)
timer = Timer()

# Step_3: Start training
for epoch in range(epochs):
    metric = Accumulator(4)
    net.train()
    for inputs, labels in train_loader:
        timer.start()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output, labels)
        loss.sum().backward()
        optimizer.step()
        # 损失爆炸，明天先验证dataloader是否出错
        # print(f'l {loss.sum()}')
        # argmax存疑
        preds = torch.argmax(output, dim=1)
        acc = torch.sum(preds==labels.data)
        metric.add(loss.sum(), acc, labels.shape[0], labels.numel())
        timer.stop()
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}')
    print(f'{metric[2] * epochs / timer.sum():.1f} examples/sec on '
          f'{str(device)}')

# Step_4: Predict
def predict(img):
    # input = train_loader.normalize_image(img).unsqueeze(0)
    input = img.to(device)
    output = net(input)
    pred = torch.argmax(output, dim=1)
    return pred.squeeze(0)

def label2img(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=device)
    pred = pred.long()
    return colormap[pred, :]

test_img, test_label = val_loader.dataset.__getitem__(0)

# plt.imshow(test_img.permute(1,2,0))
test_label_image = label2img(test_label)

pred = predict(test_img.unsqueeze(0))
pred_img = label2img(pred)

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(test_img.permute(1,2,0))
plt.subplot(1,3,2)
plt.imshow(test_label_image.cpu())
plt.subplot(1,3,3)
plt.imshow(pred_img.cpu())
plt.show()
