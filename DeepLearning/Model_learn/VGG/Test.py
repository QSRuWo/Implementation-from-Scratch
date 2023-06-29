import torch
import torch.nn as nn
import torchvision.models as models



net = models.vgg16(pretrained=False)

def init(layer):
    torch.manual_seed(42)
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.xavier_normal_(layer.weight)

net.apply(init)

print(net.state_dict().get('features.0.weight'))