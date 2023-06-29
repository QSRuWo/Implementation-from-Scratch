import torch
import torch.nn as nn
import torchvision

num_classes = 21

def build_net():
    # Get the pretrain_net and delete the last two layers.
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])

    # Build FCN
    net.add_module('last_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('Transposed_Convolution', nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16))

    # Initialize bilinear interpolation kernel
    def bilinear_kernel(in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = (torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
        filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
        weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
        weight[range(in_channels), range(out_channels), :, :] = filt
        return weight

    kernel = bilinear_kernel(num_classes, num_classes, 64)
    net.Transposed_Convolution.weight.data.copy_(kernel)

    return net

