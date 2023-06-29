import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

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


def test_crop(self):
    image = torchvision.io.read_image(self.image_paths[0], mode=torchvision.io.image.ImageReadMode.RGB)
    label = torchvision.io.read_image(self.label_paths[0], mode=torchvision.io.image.ImageReadMode.RGB)
    image, label = self.voc_rand_crop(image, label, 100, 120)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(label.permute(1, 2, 0))
    plt.show()




def voc_colormap2classes():
    colormap2classes = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2classes[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2classes

def voc_label2indices(label, colormap2classes):
    label = label.permute(1, 2, 0).numpy().astype('int32')
    idx = [(label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]]
    return colormap2classes[idx]

class VOCSegDataset(Dataset):
    def __init__(self, root, is_train, crop_size):
        super(VOCSegDataset, self).__init__()
        self.root = root
        self.crop_size = crop_size
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.path_file = os.path.join(self.root, 'ImageSets/Segmentation', 'train.txt') if is_train \
            else os.path.join(self.root, 'ImageSets/Segmentation', 'val.txt')
        self.image_root = os.path.join(self.root, 'JPEGImages')
        self.label_root = os.path.join(self.root, 'SegmentationClass')
        self.image_paths, self.label_paths = [], []
        with open(self.path_file, 'r') as f:
            for line in f:
                self.image_paths.append(os.path.join(self.image_root, line.strip() + '.jpg'))
                self.label_paths.append(os.path.join(self.label_root, line.strip() + '.png'))
        self.images, self.labels = self.read_voc_images_labels(self.image_paths, self.label_paths)
        self.images = [self.normalize_image(image) for image in self.filter(self.images)]
        self.labels = self.filter(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.voc_rand_crop(self.images[idx], self.labels[idx], *self.crop_size)
        return (image, voc_label2indices(label, voc_colormap2classes()))


    def voc_rand_crop(self, image, label, crop_height, crop_width):
        rect = transforms.RandomCrop.get_params(image, (crop_height, crop_width))
        image = F.crop(image, *rect)
        label = F.crop(label, *rect)
        return image, label

    def filter(self, images):
        return [image for image in images if (image.shape[1] >= self.crop_size[0] and image.shape[2] >= self.crop_size[1])]

    def read_voc_images_labels(self, image_paths, label_paths):
        images = []
        labels = []
        mode = torchvision.io.image.ImageReadMode.RGB
        for i in range(len(image_paths)):
            images.append(torchvision.io.read_image(image_paths[i], mode=mode))
            labels.append(torchvision.io.read_image(label_paths[i], mode=mode))
        return images, labels

    def normalize_image(self, image):
        return self.transform(image.float())

def create_voc_loader(batch_size, crop_size):
    root = r'G:\DeepLearningDataset\VOC2012\VOCdevkit\VOC2012'
    train_dataset = VOCSegDataset(root, is_train=True, crop_size=crop_size)
    val_dataset = VOCSegDataset(root, is_train=False, crop_size=crop_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f'Reading train images: {len(train_loader.dataset)}\n'
          f'Reading val images: {len(val_loader.dataset)}')
    return train_loader, val_loader


if __name__ == '__main__':
    root = r'G:\DeepLearningDataset\VOC2012\VOCdevkit\VOC2012'
    train_dataset = VOCSegDataset(root, is_train=True, crop_size=(320, 480))
    print(len(train_dataset))
    val_dataset = VOCSegDataset(root, is_train=False, crop_size=(320, 480))
    print(len(val_dataset))
    # test_label = torchvision.io.read_image(r'G:\DeepLearningDataset\VOC2012\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png', mode=torchvision.io.image.ImageReadMode.RGB)
    # print(test_label.shape)
    # y = voc_label2indices(test_label, voc_colormap2classes())
    # print(y[105:115, 130:140])
    batch_size = 64
    train_iter = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True,
        num_workers=1)
    for X, Y in train_iter:
        print(X.shape)
        print(Y.shape)
        break