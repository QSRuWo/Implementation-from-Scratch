# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader
#
# # Load MNIST dataset
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
#
# train_data = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
# test_data = datasets.MNIST(root='./dataset', train=False, transform=transform)
#
# train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
#
# # Load VGG16 model
# model = models.vgg16(pretrained=True)
#
# # Freeze the features layers
# for param in model.features.parameters():
#     param.requires_grad = False
#
# # Modify the final layer to have 10 outputs (for the 10 classes of MNIST)
# model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
#
# # Define the criterion and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
#
# # Training
# model.train()
#
# for epoch in range(10):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 200 == 199:    # print every 200 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 200))
#             running_loss = 0.0
#
# print('Finished Training')
#
# # Testing
# model.eval()
#
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the test images: %d %%' % (
#     100 * correct / total))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG16 requires the input size of (224, 224)
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization values for ImageNet
])

# Total data is 60,000 for MNIST, split it into 50,000 for training and 10,000 for validation
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_data, valid_data = random_split(train_data, [50000, 10000])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)

# Load VGG16 model
model = models.vgg16(pretrained=True)

# Freeze the features layers
for param in model.features.parameters():
    param.requires_grad = False

# Modify the final layer to have 10 outputs (for the 10 classes of MNIST)
model.classifier[6] = nn.Linear(in_features=4096, out_features=10)

# Move the model to the GPU if available
model = model.to(device)

# Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# Training
model.train()

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

# Testing
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in valid_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the validation images: %d %%' % (
    100 * correct / total))

