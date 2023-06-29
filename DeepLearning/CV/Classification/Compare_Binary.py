import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
import os

data_path = "./dataset/dogs-vs-cats-small"

# # Load the data
# data_dir_train = 'path_to_your_train_data'
# data_dir_val = 'path_to_your_val_data'
# data_dir_test = 'path_to_your_test_data'

# classes = ['class1', 'class2']

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# train_dataset = torchvision.datasets.ImageFolder(root=data_dir_train, transform=data_transforms)
# val_dataset = torchvision.datasets.ImageFolder(root=data_dir_val, transform=data_transforms)
# test_dataset = torchvision.datasets.ImageFolder(root=data_dir_test, transform=data_transforms)

# create dataset
train_dataset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
val_dataset = ImageFolder(os.path.join(data_path, 'validation'), transform=transform)
test_dataset = ImageFolder(os.path.join(data_path, 'test'), transform=transform)

# Create dataloaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Load VGG16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=False)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(4096, 1)  # Adjust to output one value
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, criterion, optimizer, dataloader):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        print(f'outputs: {outputs}')
        preds = torch.round(torch.sigmoid(outputs))
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc

# Validation function
def validate_model(model, criterion, dataloader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs))
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))

    train_loss, train_acc = train_model(model, criterion, optimizer, train_dataloader)
    print('Train Loss: {:.4f}, Acc: {:.4f}'.format(train_loss, train_acc))

    val_loss, val_acc = validate_model(model, criterion, val_dataloader)
    print('Val Loss: {:.4f}, Acc: {:.4f}'.format(val_loss, val_acc))

print('Training complete')

# Test the model
test_loss, test_acc = validate_model(model, criterion, test_dataloader)
print('Test Loss: {:.4f}, Acc: {:.4f}'.format(test_loss, test_acc))

