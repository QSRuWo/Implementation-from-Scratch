import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, DataLoader, Subset
from torch.optim import SGD


# 1. Define Dataloaders
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.MNIST(root='./dataset', train=True, transform=transforms, download=True)
train_dataset, valid_dataset = random_split(train_data, [50000, 10000])
train_dataset = Subset(train_dataset, list(range(200)))
valid_dataset = Subset(valid_dataset, list(range(100)))
test_data = datasets.MNIST(root='./dataset', train=False, transform=transforms)
test_dataset = random_split(test_data, [10000])
test_dataset = Subset(test_dataset, list(range(100)))

train_loader = DataLoader(train_dataset, batch_size=16,shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# 2. Define model
model = models.vgg16(pretrained=True)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
model = model.to(device)

# 3. Define optimizer and loss function
optimizer = SGD(params=model.classifier.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 4. Train and Valid
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.
    running_acc = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        running_acc += torch.sum(preds == labels.data)

    running_loss /= len(train_loader)
    running_acc = running_acc.double() / len(train_dataset)
    print('Training Epoch {} loss: {:.4f} acc: {:.4f}'.format(epoch+1, running_loss, running_acc))

    model.eval()
    running_loss = 0.
    running_acc = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            running_acc += torch.sum(preds == labels.data)

    running_loss /= len(valid_loader)
    running_acc = running_acc.double() / len(valid_dataset)
    print('Validation loss {:.4f} acc {:.4f}'.format(running_loss, running_acc))

# 5. Test
running_loss = 0.
running_acc = 0
with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        running_acc += torch.sum(preds == labels.data)

running_loss /= len(valid_loader)
running_acc = running_acc.double() / len(valid_dataset)
print('Test loss {:.4f} acc {:.4f}'.format(running_loss, running_acc))
print('Finished Training')


