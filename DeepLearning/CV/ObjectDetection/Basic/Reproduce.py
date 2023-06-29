import torch

def multibox_prior_1(data, size, ratios):
    in_height, in_width = data.shape[-2:]
    num_boxes = len(size) + len(ratios) - 1
    device = data.device
    size_tensor = torch.tensor(size, device=device)
    ratios_tensor = torch.tensor(ratios, device=device)

    step_w = 1 / in_width
    step_h = 1 / in_height
    offset_w, offset_h = 0.5, 0.5

    center_x_range = (torch.arange(in_width, device=device) + offset_w) * step_w
    center_y_range = (torch.arange(in_height, device=device) + offset_h) * step_h
    center_y, center_x = torch.meshgrid(center_y_range, center_x_range)
    center_x, center_y = center_x.reshape(-1), center_y.reshape(-1)
    centers = torch.stack((center_x, center_y, center_x, center_y), dim=1)
    centers = centers.repeat_interleave(num_boxes, dim=0)

    w = torch.cat((size_tensor * torch.sqrt(ratios_tensor[0]), size[0] * torch.sqrt(ratios_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratios_tensor[0]), size[0] / torch.sqrt(ratios_tensor[1:])))
    manipulations = torch.stack((-w, -h, w, h), dim=0).T.repeat(in_height * in_width, 1) / 2

    output = centers + manipulations

    return output.unsqueeze(0)


X = torch.rand((1, 3, 360, 480))
size = [0.75, 0.5, 0.25]
ratio = [1, 2, 0.5]
Y = multibox_prior_1(X, size, ratio)
print(Y.shape)

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

X = torch.rand((1, 3, 360, 480))
size = [0.75, 0.5, 0.25]
ratio = [1, 2, 0.5]
Z = multibox_prior(X, size, ratio)
print(Z.shape)
print(torch.allclose(Y,Z))