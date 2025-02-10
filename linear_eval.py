import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from resnet import Model
from utils import GaussianBlur
from tqdm import tqdm
import sys

# Argument parser
parser = argparse.ArgumentParser(description='Train a classifier with a pretrained model')
parser.add_argument('--model_path', type=str, default='checkpoint_epoch_5.pth', help='Path to pretrained model')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save log file')
args = parser.parse_args()

# Create log directory if it doesn't exist
os.makedirs(args.log_dir, exist_ok=True)
log_file = os.path.join(args.log_dir, 'training_log.txt')

# Redirect stdout to both console and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    
    def write(self, message):
        if "Home device:" not in message and "Files already downloaded and verified" not in message:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Arguments: {vars(args)}")

# Define the Net class
class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()
        model = Model().to(device)
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.encoder = model.encoder  # Directly use the encoder
        self.fc = nn.Linear(512, num_class, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

def train_val(net, data_loader, train_optimizer, epoch, epochs):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_loader:
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
    
    log_message = '{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'.format(
        'Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
        total_correct_1 / total_num * 100, total_correct_5 / total_num * 100
    )
    print(log_message)
    
    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100

# Main script
if __name__ == '__main__':
    model_path = args.model_path
    batch_size = args.batch_size
    epochs = args.epochs

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

    cifar10_train = datasets.CIFAR10(root="./dataset/cifar10", train=True, download=True, transform=train_transform)
    cifar10_test = datasets.CIFAR10(root="./dataset/cifar10", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, pin_memory=False)

    model = Net(num_class=10, pretrained_path=model_path).to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()

    acc = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer, epoch, epochs)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, epoch, epochs)
        acc.append(test_acc_1)

    best_acc = max(acc)
    print(f'Best ACC@1: {best_acc:.2f}%')
