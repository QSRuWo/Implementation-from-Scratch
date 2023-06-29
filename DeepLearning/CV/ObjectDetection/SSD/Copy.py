import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import multibox_prior
from d2l import torch as d2l
import torch.optim as optim


def cls_predictor(num_inputs, num_anchors, num_classes):
    # Why num_classes + 1?
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    # Why num_anchors * 4?
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# def forward(x, block):
#     return block(x)
#
# Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
# Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
# print(f'Y1 {Y1.shape}')
# print(f'Y2 {Y2.shape}')

def flatten_pred(pred):
    # 此处permute无意义
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(pred) for pred in preds], dim=1)

# # Y1, Y2值有过更新
# print(f'concat test {concat_preds([Y1, Y2]).shape}')

def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

# print(f'down sample test {forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape}')

def base_net():
    blk = []
    channels = [3, 16, 32, 64]
    for i in range(len(channels) - 1):
        blk.append(down_sample_blk(channels[i], channels[i + 1]))
    return nn.Sequential(*blk)

# print(f'base net test {forward(torch.zeros((2, 3, 256, 256)), base_net()).shape}')

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = multibox_prior(Y, size, ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__()
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i], getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')
            )
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

net = TinySSD(num_classes=1)
input = torch.rand((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(input)
print(f'anchors {anchors.shape}')
print(f'cls_preds {cls_preds.shape}')
print(f'bbox_preds {bbox_preds.shape}')

batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

device = torch.device('cuda')
optimizer = optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_mask):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_mask, bbox_labels * bbox_mask).mean(dim=1)
    # print(f'cls shape {cls.shape}')
    # print(f'bbox shape {bbox.shape}')
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # print(f'eval cls_preds {cls_preds.argmax(dim=-1).shape}')
    # print(f'eval cls_labels {cls_labels.shape}')
    return float(
        (cls_preds.argmax(dim=-1).type(cls_preds.dtype) == cls_labels).sum()
    )

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float(
        (torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum()
    )

num_epochs, timer = 20, d2l.Timer()
# animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class_error', 'bbox_mae'])
net = net.to(device)
for epoch in range(num_epochs):
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        optimizer.zero_grad()
        features, target = features.to(device), target.to(device)
        anchors, cls_preds, bbox_preds = net(features)
        # print(f'anchors shape {anchors.shape}')
        # print(f'target shape {target.shape}')
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, target)
        # print(f'cls_preds shape {cls_preds.shape}')
        # print(f'cls labels shape {cls_labels.shape}')
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        optimizer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(), bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    # animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')
import torchvision
X = torchvision.io.read_image(r'D:\Pytorch_Practice\DeepLearning\CV\ObjectDetection\data\banana-detection\bananas_train\images\0.png').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)

def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    d2l.plt.show()

display(img, output.cpu(), threshold=0.9)