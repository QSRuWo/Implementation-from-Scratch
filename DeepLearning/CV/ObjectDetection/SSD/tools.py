import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def box_corner_to_center(boxes):
    '''
    Convert corner coordinate to center coordinate
    :param boxes:
    :return:
    '''
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x2 - x1) / 2
    cy = (y2 - y1) / 2
    w =  x2 - x1
    h = y2 - y1
    center = torch.stack((cx, cy, w, h), dim=-1)
    return center

def box_center_to_corner(boxes):
    '''
    Convert center coordinate to corner coordinate
    :param boxes:
    :return:
    '''
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    corner = torch.stack((x1, y1, x2, y2), dim=-1)
    return corner

def multibox_prior(data, sizes, ratios):
    '''
    Generate anchor boxes with different shapes centered on each pixel
    :param data: input tensor
    :param sizes: ?
    :param ratios: ?
    :return:
    '''
    # Step_1: Get all pixels' coordinate in shape of (num_pixels, 4), and this 4 means (x_center, y_center, x_center, y_center)
    in_width, in_height = data.shape[-2:]
    device = data.device
    offset_x, offset_y = 0.5, 0.5
    step_x, step_y = 1 / in_width, 1 / in_height
    # zoom x, y into [0, 1]
    x = (torch.arange(in_width, device=device) + offset_x) * step_x
    y = (torch.arange(in_height, device=device) + offset_y) * step_y
    # Get all possible pair of x_center, y_center
    center_y, center_x = torch.meshgrid(y, x)
    center_y, center_x = center_y.reshape(-1), center_x.reshape(-1)
    # Get coordinate of each pixel as (x_center, y_center, x_center, y_center)
    center_points = torch.stack((center_x, center_y, center_x, center_y), dim=1)

    # Step_2: Calculate width and height of each kind of anchors and add with center points coordinate to finally get the coordiate of anchors in shape of (1, num_anchors, 4).
    # This 4 means (upperleft_x, upperleft_y, lowerright_x, lowerright_y)
    # Formula: W = ws√r, H = hs/√r, and only consider (s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1)
    num_boxes = len(sizes) + len(ratios) - 1
    sizes_tensor = torch.tensor(sizes, device=device)
    ratios_tensor = torch.tensor(ratios, device=device)
    w = torch.cat((sizes_tensor * torch.sqrt(ratios_tensor[0]), sizes[0] * torch.sqrt(ratios_tensor[1:]))) * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((sizes_tensor / torch.sqrt(ratios_tensor[0]), sizes[0] / torch.sqrt(ratios_tensor[1:])))
    points_shift = torch.stack((-w, -h, w, h), dim=1)

    # Add center_points and points_shift
    # Every pixel will generate num_boxes anchors, so there is num_boxes way to generate different boxes.
    center_points = center_points.repeat_interleave(num_boxes, dim=0)
    points_shift = points_shift.repeat(in_width * in_height, 1)
    anchors = center_points + points_shift
    # It seems like this unsqueeze is meaningless
    return anchors.unsqueeze(0)

class Timer:
    '''
    Record Multiple Running Times
    '''
    def __init__(self):
        self.time = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.time.append(time.time() - self.tik)
        return self.time[-1]

    def avg(self):
        return sum(self.time) / len(self.time)

    def sum(self):
        return sum(self.time)

    def cumsum(self):
        return np.array(self.time).cumsum().tolist()

class Accumulator:
    '''
    For accumulating sums over n variables
    '''
    def __init__(self, n):
        self.data = [0.] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def box_iou(boxes1, boxes2):
    '''
    Calculate iou of all boxes in boxes1 to boxes2
    :param boxes1: (num_boxes1, 4)
    :param boxes2: (num_boxes2, 4)
    :return:
    '''
    # Step_1: Get coordinate of lowerleft and upperright corners of intersection area
    # If do not expand dimension here, boxes1 and boxes2 will not compare their all one by one, for example boxes1(0,) to boxes(:,),......
    # but only one to one. boxes1(0,) to boxes(0,), boxes1(1,) to boxes(1,)
    lowerleft = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    upperright = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # This is for a case if intersection area = 0
    distance = (upperright - lowerleft).clamp(min=0)
    # Step_2: Calculate intersection area
    intersection_area = distance[:, :, 0] * distance[:, :, 1]
    # Step_3: Define a lambda function to calculate area
    boxes_area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # Step_4: Calculate union area
    union_area = boxes_area(boxes1)[:, None] + boxes_area(boxes2) - intersection_area
    # Step_5: Calculate and return IoU
    # Explanation for output shape: ith row of output means ith of boxes1, jth col of output means iou with jth boxes2
    # For here, boxes 1 are anchors and boxes 2 are labels.(Assume more than one label, but actually only 1 label will be passed in.)
    # (0, 0) is iou between the first anchor and first label, (0, 1) is iou between the first anchor and second label
    # (1, 0) is the second anchor with first label, (1, 1) is iou between the second anchor and second label
    # Instance is in the main function
    return intersection_area / union_area

def assign_anchor_to_bbox(ground_truth_box, anchors, device, iou_threshold=0.5):
    '''
    Assign the nearest anchors to bbox.
    :param ground_truth_box:
    :param anchors:
    :param device:
    :param iou_threshold:
    :return:
    '''
    # Step_1: Define some parameters and get all iou pair between all anchors and bbox
    num_anchors, num_gt_box = anchors.shape[0], ground_truth_box.shape[0]
    jaccard = box_iou(anchors, ground_truth_box)
    anchors_bbox_map = torch.full((num_anchors,), -1, device=device, dtype=torch.long)

    # Step_2: Assign all anchors which iou >= threshold to bbox
    # Tips: We have mentioned before in box_iou function that row number represent which anchors, and col number represent which bbox.
    max_iou, indices = torch.max(jaccard, dim=1)
    # Get row number(Which anchor)
    anc_idx = torch.nonzero(max_iou >= iou_threshold).reshape(-1)
    # Get col number(Which bbox)
    bbox_idx = indices[max_iou >= iou_threshold]
    # Map anchor and bbox through their idx
    anchors_bbox_map[anc_idx] = bbox_idx

    # Step_3: In a case, there is no anchors iou with bbox >= threshold, then assign the anchor with the largest iou to bbox.
    # Once find one pair of anchor and bbox, then map them, and discard this pair
    row_discard = torch.full((num_gt_box,), -1)
    col_discard = torch.full((num_anchors,), -1)
    for _ in range(num_gt_box):
        # Find the idx of the max iou
        max_idx = torch.argmax(jaccard)
        # Get the row number(Which anchor)
        anc_row = (max_idx / num_gt_box).long()
        # Get the col number(Which bbox)
        box_col = (max_idx % num_gt_box).long()
        # Map
        anchors_bbox_map[anc_row] = box_col
        # Discard
        jaccard[anc_row, :] = row_discard
        jaccard[:, box_col] = col_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    '''
    This function is to calculate offsets between anchors and bbox
    :param anchors:
    :param assigned_bb:
    :param eps:
    :return:
    '''
    c_anchors = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anchors[:, :2]) / c_anchors[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anchors[:, 2:])
    offset = torch.cat((offset_xy, offset_wh), dim=1)
    return offset

def multibox_target(anchors, labels):
    '''
    This function is used to calculate bbox_offset,
    bbox_masks(Adequate anchors will be retained and others will be set to 0),
    and class_labels of each anchors
    :param anchors:
    :param labels:
    :return:
    '''
    # Step_1: Define some hyper-parameters
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_masks, batch_class_labels = [], [], []
    device, num_anchors = labels.device, anchors.shape[0]
    # Step_2: Find relationship between each label in one batch with anchors
    for i in range(batch_size):
        label = labels[i, :, :]
        # Map bbox to adequate(iou >= iou_threshold) anchors
        # Shape is (num_anchors,)
        # Its value means idx of label, more specific, 0 means assigned to label, -1 means not assigned.
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        # Get those assigned anchors indices
        indices_adequate = torch.nonzero(anchors_bbox_map >= 0)
        # First, calculate masks
        # Set adequate anchors value to 1, others to 0
        masks = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

        # Second calculate offset
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # Get bbox idx and overwrite abm
        bbox_idx = anchors_bbox_map[indices_adequate]
        assigned_bb[indices_adequate] = label[bbox_idx, 1:]
        offset = offset_boxes(anchors, assigned_bb) * masks

        # Third, calculate class labels
        # Give all anchors a class label. If adequate, 1, else 0
        class_labels = torch.zeros((num_anchors,), dtype=torch.long, device=device)
        class_labels[indices_adequate] = label[bbox_idx, 0].long() + 1

        # Integrate offsets, masks, class labels
        batch_offset.append(offset.reshape(-1))
        batch_masks.append(masks.reshape(-1))
        batch_class_labels.append(class_labels)
    # Step_4: Integrate batch_offsets, batch_masks, batch_class_labels
    bbox_offsets = torch.stack(batch_offset)
    bbox_masks = torch.stack(batch_masks)
    bbox_class_labels = torch.stack(batch_class_labels)

    return (bbox_offsets, bbox_masks, bbox_class_labels)

def offset_inverse(anchors, offset_preds):
    '''
    This function is to calculate predicted bounding box by anchors and predicted offsets
    :param anchors:
    :param offset_preds:
    :return:
    '''
    anc_c = box_corner_to_center(anchors)
    pred_xy = (offset_preds[:, :2] * anc_c[:, 2:] / 10) + anc_c[:, :2]
    pred_wh = torch.exp(offset_preds[:, 2:] / 5) * anc_c[:, 2:]
    pred_bbox = torch.cat((pred_xy, pred_wh), dim=1)
    pred_bbox = box_center_to_corner(pred_bbox)
    return pred_bbox

def NMS(boxes, scores, iou_threshold):
    '''
    This function is to implement Non-Maximum Suppression
    :param boxes:
    :param scores:
    :param iou_threshold:
    :return:
    '''
    box_indices = torch.argsort(scores, dim=-1, descending=True)
    # keep is a list of idx of predicted bounding boxes
    keep = []
    while box_indices.numel() > 0:
        idx = box_indices[0]
        keep.append(idx)
        if box_indices.numel() == 1: break
        iou = box_iou(boxes[idx, :].reshape(-1, 4),
                      boxes[box_indices[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        box_indices = box_indices[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold = 0.5, pos_threshold = 0.009999999):
    '''
    This function is to predict bounding box with using NMS
    :param cls_probs: (1, 2, 5444)
    :param offset_preds: (1, 21776)
    :param anchors: (1, 5444, 4)
    :param nms_threshold:
    :param pos_threshold:
    :return:
    '''
    # Step_1: Define some parameters
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    # Step_2: Use predictions and NMS to predict bboxes
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # Get predicted bboxes
        predicted_bb = offset_inverse(anchors, offset_pred)
        # Get confidence level and class id
        conf, class_id = torch.max(cls_prob[1:], dim=0)
        # Conduct NMS
        keep = NMS(predicted_bb, conf, nms_threshold)
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        # Get keep and non-keep indices
        combined = torch.cat((keep, all_idx), dim=0)
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts==1]
        all_idx_sorted = torch.cat((keep, non_keep))
        # Set class of all non-keep boxes to -1
        class_id[non_keep] = -1
        # Resort class_id
        class_id = class_id[all_idx_sorted]
        conf, predicted_bb = conf[all_idx_sorted], predicted_bb[all_idx_sorted]
        # Set class of those conf < pos_threshold(positive, i.e. non-background) to -1
        below_pos = (conf < pos_threshold)
        class_id[below_pos] = -1
        conf[below_pos] = 1 - conf[below_pos]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

def bbox_to_rect(bbox, color):
    '''
    This function is to define a plt rect according to bbox
    :param bbox:
    :param color:
    :return:
    '''
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1], fill=False, edgecolor=color, linewidth=2)

def show_bboxes(axes, bboxes, labels=None, colors=None):
    '''
    This function is to display bboxes on image
    :param axes:
    :param bboxes:
    :param labels:
    :param colors:
    :return:
    '''
    def make_list(obj, default=None):
        if obj == None:
            obj = default
        elif not isinstance(obj, (tuple, list)):
            obj = [obj]
        return obj
    labels = make_list(labels,)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color=color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))


if __name__ == '__main__':
    # X = torch.rand((1,3,5,10))
    # sizes = [0.75, 0.5, 0.25]
    # ratios = [1, 2, 0.5]
    # print(multibox_prior(X, sizes, ratios).shape)
    box2 = torch.tensor([[1, 2, 3, 4], [10, 11, 12, 13], [10, 11, 12, 13], [10, 11, 12, 13]])
    box1 = torch.tensor([[0, 1, 2, 6]])
    print(assign_anchor_to_bbox(box1, box2, 'cpu'))