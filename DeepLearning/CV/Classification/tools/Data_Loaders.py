import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import ImageFolder
import PIL.Image as Image
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def create_data_loader(path, batch_size, num_workers, mean_values, std_values):
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'validation')
    test_path = os.path.join(path, 'test')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean_values, std = std_values)
    ])

    # train_dataset = ImageFolder(train_path, transform=transform)
    # val_dataset = ImageFolder(val_path, transform=transform)
    # test_dataset = ImageFolder(test_path, transform=transform)

    train_dataset = dog_vs_cat_dataset(train_path, transform=transform)
    val_dataset = dog_vs_cat_dataset(val_path, transform=transform)
    test_dataset = dog_vs_cat_dataset(test_path, transform=transform)

    print(f"train len: {len(train_dataset)}; val len: {len(val_dataset)}; test len: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader

class dog_vs_cat_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_path = []
        self.label = []

        for label, category in enumerate(['cat', 'dog']):
            category_dir = os.path.join(root_dir, category)
            for file_name in os.listdir(category_dir):
                self.img_path.append(os.path.join(category_dir, file_name))
                self.label.append(label)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        label = self.label[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

def calculate_mean_std(path):
    mean_values = np.zeros(3)
    std_values = np.zeros(3)
    num_files = 0

    for root, dir, files in os.walk(path):
        for file in files:
            img_path = os.path.join(root, file)
            img = Image.open(img_path).resize((224, 224))
            # The shape of np array converted from PIL image is (H, W, C)
            img_np = np.asarray(img, dtype=np.float32) / 255.
            mean_values += img_np.mean(axis=(0, 1))
            std_values += img_np.std(axis=(0,1))
            num_files += 1

    mean_values /= num_files
    std_values /= num_files

    return mean_values, std_values




if __name__ == '__main__':
    path = "../dataset/dogs-vs-cats-small"
    mean = [0.49, 0.45, 0.41]
    std = [0.23, 0.22, 0.22]
    # train_path = "../dataset/dogs-vs-cats-small/train"
    # mean, std = calculate_mean_std(train_path)
    # print(mean)
    # print(std)
    create_data_loader(path, 16, 0, mean, std)