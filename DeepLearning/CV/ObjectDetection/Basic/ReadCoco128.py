import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

class CoCo128_Dataset(Dataset):
    def __init__(self, root, transform=None):
        super(CoCo128_Dataset, self).__init__()
        self.root = root
        self.image_paths = glob.glob(os.path.join(root, 'images/train2017/*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        target_path = img_path.replace('.jpg', '.txt')
        target_path = target_path.replace('images', 'labels')
        target = torch.from_numpy(np.loadtxt(target_path))
        target_modified = torch.zeros([10, 5])
        if len(target.shape) == 1:
            target = target.unsqueeze(0)
        if target.shape[0] < target_modified.shape[0]:
            target_modified[:target.shape[0]] = target[0]
        else:
            target_modified[:] = target[:10]

        return img, target_modified

root = '../datasets/coco128'

transfrom = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor()
])

train_dataset = CoCo128_Dataset(root,transfrom)
# img, target = train_dataset.__getitem__(21)
img, target = train_dataset.__getitem__(0)
# print(img.shape)
# print(target.shape)
# print(target)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#
# for inputs, labels in train_loader:
#     print(inputs.shape)
#     print(labels.shape)
from torchvision.transforms.functional import to_pil_image
img = to_pil_image(img)
fig = plt.imshow(img)
f = plt.Rectangle(xy=(target[0,1]*224, target[0,2]*224), width=target[0,3]*224-target[0,1]*224, height=target[0,4]*224-target[0,1]*224, fill=False, edgecolor='blue', linewidth=2)
fig.axes.add_patch(f)
plt.show()