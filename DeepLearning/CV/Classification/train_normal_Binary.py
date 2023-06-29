import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import PIL.Image as Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from tools.callbacks import *

# 1. Define data path and device
data_path = "./dataset/dogs-vs-cats-small"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define a Callback function
save_model_callback = SaveModelCallbacks()


# 2. Create Dataset for training, validating and testing
# define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# create dataset
train_dataset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
val_dataset = ImageFolder(os.path.join(data_path, 'validation'), transform=transform)
test_dataset = ImageFolder(os.path.join(data_path, 'test'), transform=transform)

# 3. Create dataloader
# define some hyp-parameters
batch_size = 16
num_workers = 0

# create dateloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# 4. create model
# define binary classification
num_classes = 1
model = models.vgg16(pretrained=True)
for name, param in model.named_parameters():
    # print(name)
    param.requires_grad = False
model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
model.to(device)
print(model)

# 5. training and validation
# define some hyp-parameters
criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.
    running_acc = 0.
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.float().view(-1, 1).to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = torch.round(torch.sigmoid(outputs))

        running_acc += torch.sum(preds == labels.data)

    running_loss /= len(train_loader)
    print(f'Training Epoch {epoch+1} loss is {running_loss}')
    running_acc = running_acc.double() / len(train_dataset)
    print(f'Training Epoch {epoch+1} acc is {running_acc}')

    if epoch==0:
        save_model_callback.on_epoch_end(model, epoch)

    model.eval()
    running_loss = 0.
    running_acc = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().view(-1, 1).to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.round(torch.sigmoid(outputs))

            running_acc += torch.sum(preds == labels.data)
    running_loss /= len(val_loader)
    running_acc = running_acc.double() / len(val_dataset)

    print(f'Validating loss is {running_loss}')
    print(f'Validating acc is {running_acc}')

# 6. Testing
test_loss = 0.
test_acc = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.float().view(-1, 1).to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        test_loss += loss.item()

        preds = torch.round(torch.sigmoid(outputs))

        test_acc += torch.sum(preds == labels.data)

test_loss /= len(test_loader)
test_acc = test_acc.double() / len(test_dataset)

print('Test Loss: {:.4f} ACC: {:.4f}'.format(test_loss, test_acc))



print("Finished Training")