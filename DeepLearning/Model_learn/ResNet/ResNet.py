import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

# 自己实现
# class Residual(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, use_1x1_conv=False):
#         super(Residual, self).__init__()
#         self.use_1x1_conv = use_1x1_conv
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#         if use_1x1_conv:
#             self.conv_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride)
#
#     def forward(self, x):
#         y = self.conv_1(x)
#         y = self.conv_2(y)
#         if self.use_1x1_conv:
#             x = self.conv_3(x)
#         y += x
#         return F.relu(y)
#
# class ResNet_34(nn.Module):
#     def __init__(self):
#         super(ResNet_34, self).__init__()
#         self.block_1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=(7, 7), stride=2, padding=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.block_2 = nn.Sequential(
#             Residual(64, 64),
#             Residual(64, 64),
#             Residual(64, 64),
#         )
#         self.block_3 = nn.Sequential(
#             Residual(64, 128, stride=2, use_1x1_conv=True),
#             Residual(128, 128),
#             Residual(128, 128),
#             Residual(128, 128)
#         )
#         self.block_4 = nn.Sequential(
#             Residual(128, 256, stride=2, use_1x1_conv=True),
#             Residual(256, 256),
#             Residual(256, 256),
#             Residual(256, 256),
#             Residual(256, 256),
#             Residual(256, 256)
#         )
#         self.block_5 = nn.Sequential(
#             Residual(256, 512, stride=2, use_1x1_conv=True),
#             Residual(512, 512),
#             Residual(512, 512)
#         )
#         self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.linear = nn.Linear(512, 10)
#
#     def forward(self, x):
#         x = self.block_1(x)
#         x = self.block_2(x)
#         x = self.block_3(x)
#         x = self.block_4(x)
#         x = self.block_5(x)
#         x = self.global_avgpool(x)
#         x = torch.flatten(x)
#         x = self.linear(x)
#
#         return x
#
# net = ResNet_34()
# from torchsummary import summary
# device = torch.device('cuda')
# net.to(device)
# summary(net, (1,224,224))

# 沐神版本
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True,
                         strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))

device = torch.device('cuda')
net.to(device)
summary(net, (1,224,224))