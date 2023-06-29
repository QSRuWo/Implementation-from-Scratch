import torch
from ReadData import create_banana_dataloader
from Model import TinySSD
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
from tools import Timer, Accumulator, multibox_target, multibox_detection, show_bboxes
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# Step_1: Create train and val dataloader
batch_size = 32
train_loader, val_loader = create_banana_dataloader(batch_size)

# Step_2: Define SSD net and hyperparameters
device = torch.device('cuda')
net = TinySSD(1)
net = net.to(device)
summary(net, (3,224,224))

optimizer = optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
cls_criterion = nn.CrossEntropyLoss(reduction='none')
bbox_criterion = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    '''
    Calculate loss
    :param cls_preds:
    :param cls_labels:
    :param bbox_preds:
    :param bbox_labels:
    :param bbox_masks:
    :return:
    '''
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    # Calculate classification loss
    cls_loss = cls_criterion(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # Calculate bounding box loss
    # Still have question here why calculate like this and what is bbox_masks.
    bbox_loss = bbox_criterion(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls_loss + bbox_loss

def cls_acc(cls_preds, cls_labels):
    '''
    Calculate the accuracy of classification
    :param cls_preds:
    :param cls_labels:
    :return:
    '''
    return float((cls_preds.argmax(dim=-1).type(cls_preds.dtype) == cls_labels).sum())

def bbox_acc(bbox_preds, bbox_labels, bbox_masks):
    '''
    Calculate accuracy of bbox
    :param bbox_preds:
    :param bbox_labels:
    :param bbox_masks:
    :return:
    '''
    return float((torch.abs(bbox_labels - bbox_preds) * bbox_masks).sum())

epochs = 1
Timer = Timer()

# Start training
for epoch in range(epochs):
    # Create a list maybe for recording the results?
    metric = Accumulator(4)
    net.train()
    for input, labels in train_loader:
        Timer.start()
        input, labels = input.to(device), labels.to(device)
        # Zero all the previous gradient
        optimizer.zero_grad()
        # Get input through net work
        # Here we do not return feature map.
        anchors, cls_preds, bbox_preds = net(input)
        # Since shape of labels is (batch_size, 1, 5), all class label and bbox labels are contained in this 5. We have to separate them.
        # bbox_labels: bbox offsets
        # Step into this function to check what these 3 variables mean.
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, labels)
        # Calculate loss
        loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        # print(f'cls labels {cls_labels.shape}')
        loss.mean().backward()
        optimizer.step()
        metric.add(cls_acc(cls_preds, cls_labels), cls_labels.numel(), bbox_acc(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
        # print(f'metric {metric.data}')
    cls_err = 1 - metric[0] / metric[1]
    bbox_mae = metric[2] / metric[3]
print(f'class err {cls_err:.2e}; bbox mae {bbox_mae:.2e}')
print(f'{len(train_loader.dataset) / Timer.stop():.1f} examples/sec on {str(device)}')

# Predicting
# Here we only predict one picture
def predict(val_data):
    net.eval()
    anchors, cls_preds, bbox_preds = net(val_data.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    out = multibox_detection(cls_probs, bbox_preds, anchors)
    print(out.shape)
    idx = [i for i, row in enumerate(out[0]) if row[0] != -1]
    print(f'idx shape {len(idx)}')
    print(out[0, idx])
    return out[0, idx]

# val_data = torchvision.io.read_image(r'D:\Pytorch_Practice\DeepLearning\CV\ObjectDetection\data\banana-detection\bananas_val\images\0.png').unsqueeze(0).float()
val_data = torchvision.io.read_image(r'D:\Pytorch_Practice\DeepLearning\CV\ObjectDetection\data\banana-detection\bananas_train\images\0.png').unsqueeze(0).float()
prediction = predict(val_data)

def display(image, prediction, threshold):
    fig = plt.imshow(image)
    for row in prediction:
        score = float(row[1])
        if score < threshold:
            continue
        w, h = image.shape[:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    plt.show()

image = val_data.squeeze(0).permute(1, 2, 0).long()
print(f'prediction shape {prediction.shape}')
display(image, prediction.cpu(), threshold=0.9)