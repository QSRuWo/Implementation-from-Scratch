import torch
import PIL.Image as Image
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt

def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# The logic is to calculate all the center coordinate(every pixel), and w, h of each kind of boxes according to
# W = ws√r, H = hs/√r, and only consider (s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1)
# Then finally get result which shape like (1, height * width * box_number, 4)
# And this 4 is (left_up_corner_x, left_up_corner_y, right_down_corner_x, right_down_corner_y)
def multibox_prior(data, size, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(size), len(ratios)
    # Only Consider (s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(size, device=device)
    ratios_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = 0.5, 0.5
    step_h = 1 / in_height
    step_w = 1 / in_width

    # narrowed all coordinate into [0, 1]
    center_h = (torch.arange(in_height, device=device) + offset_h) * step_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * step_w
    # Get all x, y combination
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # The width and height of anchors
    # W = ws√r, H = hs/√r, and only consider (s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1)
    # Here is to compute s√r?
    w = torch.cat((size_tensor * torch.sqrt(ratios_tensor[0]), size[0] * torch.sqrt(ratios_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratios_tensor[0]), size[0] / torch.sqrt(ratios_tensor[1:])))
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    out_grid = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = anchor_manipulations + out_grid

    return output.unsqueeze(0)

img = Image.open(r'D:\Pytorch_Practice\559_Final_Project\datasets\coco128\images\train2017\000000000009.jpg')

img_tensor = to_tensor(img)
print(img_tensor.shape)

Y = multibox_prior(img_tensor, size=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)
Y = Y.reshape((480, 640, 5, 4))

def bbox_to_rect(bbox, color):
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=(bbox[2] - bbox[0]), height=(bbox[3] - bbox[1]), fill=False, edgecolor=color, linewidth=2)

def show_boxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_value=None):
        if obj is None:
            obj = default_value
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[(i % len(colors))]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))

fig = plt.imshow(img)
box_scale = torch.tensor((640, 480, 640, 480))
bboxes = Y[250, 250, :, :] * box_scale
labels = ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5']
show_boxes(fig.axes, bboxes, labels)
plt.show()

def boxes_iou(boxes_1, boxes_2):
    boxes_area = lambda boxes: (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    boxes_1_area = boxes_area(boxes_1)
    boxes_2_area = boxes_area(boxes_2)
    inter_lowerleft = torch.max(boxes_1[:, None, :2], boxes_2[:, :2])
    inter_upperright = torch.min(boxes_1[:, None, 2:], boxes_2[:, 2:])
    inters = (inter_upperright - inter_lowerleft).clamp(min=0)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = boxes_1_area[:, None] + boxes_2_area - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth_boxes, anchors, device, iou_threshold=0.5):
    num_anchors, num_gt = anchors.shape[0], ground_truth_boxes.shape[0]
    jaccard = boxes_iou(ground_truth_boxes, anchors)
    anchors_bbox_max = torch.full((num_anchors,), fill_value=-1, dtype=torch.long, device=device)
    max_iou, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_iou >= iou_threshold).reshape(-1)
    box_j = indices[max_iou >= iou_threshold]
    anchors_bbox_max[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    raw_discard = torch.full((num_gt,), -1)
    for _ in range(num_gt):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt).long()
        anc_idx = (max_idx / num_gt).long()
        anchors_bbox_max[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = raw_discard

    return anchors_bbox_max

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换。"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1], anchors, device=device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)



ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])

fig = plt.imshow(img)
# show_boxes(fig.axes, ground_truth[:, 1:] * box_scale, ['dog', 'cat'], 'k')
# plt.show()
show_boxes(fig.axes, anchors * box_scale, ['0', '1', '2', '3', '4'])
plt.show()

labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框。"""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = boxes_iou(boxes[i, :].reshape(-1, 4), boxes[B[1]:, :].reshape).reshape(-1)
        inds = torch.nonzero((iou <= iou_threshold)).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        nonkeep = uniques[counts==1]
        all_id_sorted = torch.cat((keep, nonkeep))
        class_id[nonkeep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
        return torch.stack(out)