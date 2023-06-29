import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
import pandas as pd

root = r'D:\Pytorch_Practice\DeepLearning\CV\ObjectDetection\data\banana-detection'

def create_banana_dataloader(batch_size):
    print('Reading data from Banana dataset......')
    train_loader = DataLoader(Banana_Dataset(root, is_train=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Banana_Dataset(root, is_train=False), batch_size=batch_size, shuffle=False)
    print(f'Loaded training images {len(train_loader.dataset)}')
    print(f'Loaded validation images {len(val_loader.dataset)}')
    return train_loader, val_loader

class Banana_Dataset(Dataset):
    def __init__(self, root, is_train=True):
        super(Banana_Dataset, self).__init__()
        self.root = root
        self.data_path = os.path.join(self.root, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
        self.csv_data = pd.read_csv(self.data_path)
        self.csv_data = self.csv_data.set_index('img_name')
        self.images, self.targets = [], []
        for img_name, target in self.csv_data.iterrows():
            self.images.append(os.path.join(self.root, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}'))
            self.targets.append(list(target))
        self.targets = torch.tensor(self.targets).unsqueeze(1) / 256

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.images[idx])

        target = torch.tensor(self.targets[idx])

        return img.float(), target

if __name__ == '__main__':
    root = r'D:\Pytorch_Practice\DeepLearning\CV\ObjectDetection\data\banana-detection'
    train_dataset = Banana_Dataset(root=root, is_train=True)
    print(len(train_dataset))
    img, label = train_dataset.__getitem__(0)
    print(img.shape)
    print(label)