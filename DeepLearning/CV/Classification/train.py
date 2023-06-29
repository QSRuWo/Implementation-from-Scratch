import argparse
from tools.Data_Loaders import create_data_loader
from model.vgg16 import VGG16
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
from torchvision import models

def train(opt):
    data_path, batch_size, lr, epochs, num_workers = opt.data_path, opt.batch_size, opt.lr, opt.epochs, opt.num_workers
    train_loader, val_loader, test_loader = create_data_loader(data_path, batch_size, num_workers, opt.mean_values, opt.std_values)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = VGG16()
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # print(images)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        training_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        running_loss = 0.
        running_acc = 0.
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # print(f"outputs shape{outputs.shape}")
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                # print(f"preds shape: {preds.shape} labels.data shape {labels.data.shape}")
                running_acc  += torch.sum(preds == labels.data)
            val_loss = running_loss / len(val_loader)
            val_acc = running_acc / (len(val_loader) * batch_size)

        print(f"Epochs: {epoch+1} / {epochs}")
        print(f"Training loss: {training_loss}")
        print(f"Val loss: {val_loss}")
        print(f"Val acc: {val_acc}")

    print("====================Finished training====================")

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size for training')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate for the optimizer')
    parser.add_argument('--data_path', type=str, default="./dataset/dogs-vs-cats-small", help='Path to data')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers to read data')
    # parser.add_argument('--mean_values', type=float, default=[0.49, 0.45, 0.41], help='Mean values for Normalization')
    # parser.add_argument('--std_values', type=float, default=[0.23, 0.22, 0.22], help='Standard difference for Normalization')
    parser.add_argument('--mean_values', type=float, default=[0.485, 0.456, 0.406], help='Mean values for Normalization')
    parser.add_argument('--std_values', type=float, default=[0.229, 0.224, 0.225], help='Standard difference for Normalization')
    return parser.parse_args()

def main(opt):
    train(opt)



if __name__ == "__main__":
    opt = parse_opt()
    main(opt)