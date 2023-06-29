import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a custom dataset class
class DogCatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            for img in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, img))
                self.labels.append(0 if folder == "cat" else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

# Load data and create DataLoaders
train_dataset = DogCatDataset("./dataset/dogs-vs-cats-small/train", transform=data_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

val_dataset = DogCatDataset("./dataset/dogs-vs-cats-small/validation", transform=data_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained VGG16 model
vgg16 = models.vgg16(pretrained=False)

# Modify the model for binary classification
vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=1)
vgg16 = vgg16.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(vgg16.parameters(), lr=0.0001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for phase in ["train", "val"]:
        if phase == "train":
            vgg16.train()
            dataloader = train_dataloader
        else:
            vgg16.eval()
            dataloader = val_dataloader

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            print(inputs)
            inputs = inputs.to(device)
            labels = labels.float().view(-1, 1).to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = vgg16(inputs)
                preds = (outputs > 0).float()
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print()

    print("Training complete")

