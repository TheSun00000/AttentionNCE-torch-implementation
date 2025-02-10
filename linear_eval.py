import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
import os
from resnet import Model
from random import sample
from utils import GaussianBlur

# Argument parser
parser = argparse.ArgumentParser(description='Train a classifier with a pretrained model')
parser.add_argument('--model_path', type=str,default='checkpoint_epoch_20.pth' , help='Path to pretrained model')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Home device: {device}")

# Define the Net class
class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()
        # encoder
        model = Model().to(device)
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.encoder = model.encoder  # Directly use the encoder
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# Train or test for one epoch
def train_val(net, data_loader, train_optimizer, epoch, epochs):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100

# Main script
if __name__ == '__main__':
    # Configuration
    model_path = args.model_path
    batch_size = args.batch_size
    epochs = args.epochs

    # Transformations for CIFAR-10
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    # Load CIFAR-10 dataset
    cifar10_train = datasets.CIFAR10(root="./dataset/cifar10", train=True, download=True, transform=train_transform)
    cifar10_test = datasets.CIFAR10(root="./dataset/cifar10", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, pin_memory=False)

    # Initialize model
    model = Net(num_class=10, pretrained_path=model_path).to(device)

    # Freeze encoder parameters
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Define optimizer and loss criterion
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()

    # Training and evaluation
    acc = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer, epoch, epochs)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, epoch, epochs)
        acc.append(test_acc_1)

    print(f'Best ACC@1: {max(acc):.2f}%')
