import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import multibox_prior
from d2l import torch as d2l
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from tools import show_bboxes

def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes.

    Defined in :numref:`sec_anchor`"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes.

    Defined in :numref:`sec_anchor`"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # print(f'jaccard shape {jaccard.shape}')
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    # print(f'max iou shape {max_ious}')
    # print(f'indices shape {indices.shape}')
    # print(f'anc i before reshape {torch.nonzero(max_ious >= 0.5)}')
    anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
    # print(f'anc i shape {anc_i.shape}')
    box_j = indices[max_ious >= 0.5]
    # print(f'box j shape {box_j.shape}')
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    # print(f'num gt {num_gt_boxes}')
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        # print(f'********{anchors_bbox_map[anc_idx]}')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    # print(f'anchors_bbox_map {torch.nonzero(anchors_bbox_map>=0)}')
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    # print(f'++++{anchors.shape}')
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        # bbox_mask shape (num_anchors, 4), all its values are either 1. or 0.
        # This is for eliminate all those iou <= threshold
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        # ???
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        # Only get those pair that abm>=0
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


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

    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    return float(
        (cls_preds.argmax(dim=-1).type(cls_preds.dtype) == cls_labels).sum()
    )

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float(
        (torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum()
    )

num_epochs, timer = 30, d2l.Timer()
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
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, target)
        # print(f'cls_preds shape {cls_preds.shape}')
        # print(f'cls labels shape {cls_labels.shape}')
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        optimizer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(), bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')

# X = torchvision.io.read_image(r'D:\Pytorch_Practice\DeepLearning\CV\ObjectDetection\data\banana-detection\bananas_val\images\0.png').unsqueeze(0).float()
X = torchvision.io.read_image(r'D:\Pytorch_Practice\DeepLearning\CV\ObjectDetection\data\banana-detection\bananas_train\images\0.png').unsqueeze(0).float()
print(f'X.shape {X.shape}')
img = X.squeeze(0).permute(1, 2, 0).long()

def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes.

    Defined in :numref:`subsec_predicting-bounding-boxes-nms`"""
    print(f'boxes shape {boxes.shape}'
          f'{boxes[0, :].shape}')
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        print((boxes[i, :].reshape(-1, 4)).shape)
        print((boxes[B[1:], :].reshape(-1, 4)).shape)
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        print(f'iou shape {iou.shape}')
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        print(f'**** {(iou <= iou_threshold).shape}')
        print(f'B {B.shape}')
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression.

    Defined in :numref:`subsec_predicting-bounding-boxes-nms`
    cls_probs: (1, 2, 5444)
    offset_preds: (1, 21776)
    anchors: (1, 5444, 4)
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    print(f'cls_preds shape {cls_preds.shape}')
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    print(f'cls_probs.shape {cls_probs.shape}')
    print(f'bbox_preds.shape {bbox_preds.shape}')
    print(f'anchors.shape {anchors.shape}')
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    print(f'output.shape {output.shape}')
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    print(f'+++++++++++++ idx {len(idx)}')
    return output[0, idx]

output = predict(X)

# def display(img, output, threshold):
#     d2l.set_figsize((5, 5))
#     fig = d2l.plt.imshow(img)
#     for row in output:
#         score = float(row[1])
#         if score < threshold:
#             continue
#         h, w = img.shape[0:2]
#         bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
#         d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
#     plt.show()
def display(image, prediction, threshold):
    fig = plt.imshow(image)
    for row in prediction:
        print(f'row {row}')
        score = float(row[1])
        if score < threshold:
            continue
        w, h = image.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        print(f'bbox {bbox}')
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    plt.show()
print(f'output {output.shape}')
display(img, output.cpu(), threshold=0.9)