import torch
import torchvision
import os

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注。"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'),
                mode))
    return features, labels

voc_dir = r'G:\DeepLearningDataset\VOC2012\VOCdevkit\VOC2012'

train_features, train_labels = read_voc_images(voc_dir, True)

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射。"""
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    print(f'colormap2label.shape {colormap2label.shape}')
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引。"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 +
           colormap[:, :, 2])
    print(f'idx {idx.shape}')
    print(f'c2l {colormap2label.shape}')
    print(f'colormap2label[idx].shape {colormap2label[idx].shape}')
    return colormap2label[idx]

# label = r'G:\DeepLearningDataset\VOC2012\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png'
# label = torchvision.io.read_image(label)
# print(label.shape)
print(train_labels[0].shape)
y = voc_label_indices(train_labels[0], voc_colormap2label())
# y[105:115, 130:140], VOC_CLASSES[1]
print(y.shape)

