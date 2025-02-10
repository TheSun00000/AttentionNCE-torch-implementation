import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
from torch.utils.data import DataLoader, Dataset
from resnet import Model
from tqdm import tqdm
from evaluation import knn_evaluation
from utils import GaussianBlur

def parse_args():
    parser = argparse.ArgumentParser(description="Train SimCLR model with CLI options")
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--n_eval_epochs', type=int, default=5, help='Evaluation frequency')
    parser.add_argument('--n_save_epochs', type=int, default=20, help='Checkpoint save frequency')
    return parser.parse_args()



# Dataset class
class SimCLRDataset(Dataset):
    def __init__(self, dataset, transform, num_positives):
        self.dataset = dataset
        self.transform = transform
        self.num_positives = num_positives

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        anchor1 = self.transform(img)
        anchor2 = self.transform(img)
        positives = [self.transform(img) for _ in range(self.num_positives)]
        return [anchor1, anchor2], positives, label

# Loss function
def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return torch.cat((negative_mask, negative_mask), 0)

def AttentioNCE(out_1, out_2, out_pos, batch_size, temperature=0.5, d_pos=1., d_neg=1.):
    feature_dim = out_1.shape[1]
    num_pos = len(out_pos)
    anchor = torch.cat([out_1, out_2], dim=0)
    pos_views = torch.cat(out_pos, dim=1).view(-1, num_pos, feature_dim)
    pos_sim = torch.sum(anchor.view(2, batch_size, 1, feature_dim) * pos_views, dim=-1).view(-1, num_pos)
    alpha = F.softmax(pos_sim.detach() / d_pos, dim=-1)
    pos_score = torch.exp(torch.sum(pos_sim * alpha, dim=-1) / temperature)
    
    sim = torch.mm(anchor, anchor.t())
    mask = get_negative_mask(batch_size).to(device)
    neg_sim = sim.masked_select(mask).view(2 * batch_size, -1)
    beta = (F.softmax(neg_sim.detach() / d_neg, dim=-1) + 1e-6) * (2 * batch_size - 2)
    neg_score = torch.exp(neg_sim * beta / temperature)
    
    return -torch.log(pos_score / (pos_score + neg_score.sum(dim=-1))).mean()

def load_latest_checkpoint(model, optimizer, save_dir):
    checkpoint_files = sorted([f for f in os.listdir(save_dir) if f.endswith('.pth')],
                              key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if not checkpoint_files:
        print("No checkpoints found. Starting training from scratch.")
        return 0, [], []
    checkpoint = torch.load(os.path.join(save_dir, checkpoint_files[-1]))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Resumed training from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"], checkpoint["losses"], checkpoint["accs"]

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    cifar10_train = datasets.CIFAR10(root="./dataset/cifar10", train=True, download=True)
    simclr_dataset = SimCLRDataset(cifar10_train, transform, num_positives=4)
    data_loader = DataLoader(simclr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    encoder = Model().to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-6)
    start_epoch, losses, accs = load_latest_checkpoint(encoder, optimizer, args.save_dir)
    
    for epoch in range(start_epoch, args.epochs):
        for anchors, positives, _ in tqdm(data_loader, desc=f"Epoch {epoch}"):
            anchor1, anchor2 = anchors[0].to(device), anchors[1].to(device)
            out_1, out_2 = encoder(anchor1), encoder(anchor2)
            out_pos = [encoder(pos.to(device)) for pos in positives]
            
            loss = AttentioNCE(out_1, out_2, out_pos, batch_size=args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        if (epoch + 1) % args.n_eval_epochs == 0:
            accs.append(knn_evaluation(encoder, dataset='cifar10'))
        if (epoch + 1) % args.n_save_epochs == 0:
            torch.save({"epoch": epoch + 1, "model_state_dict": encoder.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "losses": losses, "accs": accs}, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

