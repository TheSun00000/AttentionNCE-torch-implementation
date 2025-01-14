
# imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
from evaluation import knn_evaluation
import os
from torchvision.models import resnet18
from PIL import Image

from torchvision.datasets import CIFAR10, CIFAR100
from random import sample
import cv2

import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'




# training parameters

n_pos = 4
projection_dim = 128
n_anchors = 128 # 256
epochs = 400

save_dir = "checkpointschanged_model_transfom"
n_eval_epochs = 5  # Frequency of evaluation
n_save_epochs = 20  # Frequency of saving checkpoints
lr = 1e-3  # Learning rate
weight_decay = 1e-6  # Weight decay


# Datasets


class SimCLRDataset(Dataset):
    def __init__(self, dataset, transform, num_positives):
        """
        Args:
            dataset: Base dataset (e.g., CIFAR-10)
            transform: Transformation pipeline for data augmentation
            num_positives: Number of positive samples to generate per anchor
        """
        self.dataset = dataset
        self.transform = transform
        self.num_positives = num_positives

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the anchor image and its label
        img, label = self.dataset[idx]
        
        anchor1 = self.transform(img)
        anchor2 = self.transform(img)

        # Apply transformations to generate positive samples
        positives = [self.transform(img) for _ in range(self.num_positives)]

        return [anchor1, anchor2], positives, label

# Transformation pipeline for data augmentation

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


# Loss

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask



def AttentioNCE(out_1, out_2, out_pos, batch_size, temperature=0.5, d_pos=1., d_neg=1.):
    '''
    Parameters
    ----------
    # anchor points
    out_1 : anchor features [bs,dim]
    out_2 : anchor features [bs,dim]

    # 4 positive views for  each anchor point 
    out_pos : list of positive views [num_pos,bs,dim]

    batch_size : Under SimCLR framework, negative sample size N = 2 x (batch size - 1)
    temperature: temperature scaling
    d_pos : positive scaling factor
    d_neg : negative scaling factor
    Returns: AttentioNCE loss values [2bs,]
    -------
    '''
    
    feature_dim = 128
    num_pos = len(out_pos)
    
    anchor = torch.cat([out_1, out_2], dim=0)  # [2bs*dim]

    # pos score
    pos_views = torch.cat(out_pos,dim=1).view(-1, num_pos, feature_dim) # [bs,num_pos,dim]
    pos_sim = torch.sum(anchor.view(2, batch_size, 1, feature_dim) * pos_views, dim=-1).view(-1, num_pos)
    # [2,bs,1,dim] x [bs,num_pos,dim]  -> [2,bs,num_pos,dim] ->  [2,bs,num_pos]-> [2bs,num_pos]
    alpha = torch.nn.functional.softmax(pos_sim.detach()/ d_pos, dim=-1) # [2bs,num_pos]
    pos_score = torch.exp(torch.sum(pos_sim * alpha, dim=-1) / temperature)  # [2bs,] This step is obtained based on the additivity property of inner product operations.

    # neg score
    sim = torch.mm(anchor, anchor.t().contiguous())  # [2bs,2bs]
    mask = get_negative_mask(batch_size).to(device)
    neg_sim = sim.masked_select(mask).view(2 * batch_size, -1)  # [2bs, 2bs-2]
    beta = (torch.nn.functional.softmax(neg_sim.detach() / d_neg, dim=-1) + 1e-6) * (2 * batch_size - 2)
    neg_score = torch.exp(neg_sim * beta / temperature)  # [2bs, 2N-2] This step is obtained based on the additivity property of inner product operations.

    # contrastive loss
    loss = - torch.log(pos_score / (pos_score + neg_score.sum(dim=-1))).mean() # [2bs,]
    return loss



# Model 


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.encoder = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Customize conv1
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.encoder.append(module)
        # Encoder
        self.encoder = nn.Sequential(*self.encoder)
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256, bias=False),  # ResNet18's output dimension is 512
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim, bias=True)
        )

    def forward(self, x):
        x = self.encoder(x)  # Pass through the encoder
        feature = torch.flatten(x, start_dim=1)  # Flatten features
        projections = self.projection_head(feature)  # Pass through the projection head
        projections=F.normalize(projections, dim=-1)
        return  projections
    

















# Function to find the latest checkpoint
def load_latest_checkpoint(model, optimizer, save_dir):
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        print("No checkpoints found. Starting training from scratch.")
        return 0, [], []  # Start from scratch
    
    # Sort files by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = checkpoint_files[-1]
    checkpoint_path = os.path.join(save_dir, latest_checkpoint)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    losses = checkpoint["losses"]
    accs = checkpoint["accs"]
    
    print(f"Resumed training from epoch {start_epoch} using {checkpoint_path}")
    return start_epoch, losses, accs


def main(args):
    
    n_pos = args.n_pos
    projection_dim = args.projection_dim
    n_anchors = args.n_anchors
    epochs = args.epochs

    save_dir = args.save_dir
    n_eval_epochs = args.n_eval_epochs
    n_save_epochs = args.n_save_epochs
    lr = args.lr
    weight_decay = args.weight_decay = 1e-6
    

    # Create directory for saving checkpoints
    os.makedirs(save_dir, exist_ok=True)

    # Loading the dataste and prepearing dataloader 
    cifar10_train = datasets.CIFAR10(
        root="./dataset/cifar10",
        train=True,
        download=True,
    )
    simclr_dataset = SimCLRDataset(cifar10_train, transform, n_pos)
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    data_loader = DataLoader(simclr_dataset, batch_size=n_anchors, shuffle=True, drop_last=True, generator=generator)


    # Initialize model and optimizer
    encoder = Model().to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    # Attempt to load the latest checkpoint
    start_epoch, losses, accs = load_latest_checkpoint(encoder, optimizer, save_dir)
    
    ### Main train loop
    for epoch in range(start_epoch, epochs):
        for n_step, (anchors, positives, labels) in tqdm(enumerate(data_loader), desc=f'Epoch: {epoch}', total=len(data_loader)):
            anchor1, anchor2 = anchors
            
            
            out_1 = encoder(anchor1.to(device))
            out_2 = encoder(anchor2.to(device))
            out_pos = []
            for pos in positives:
                out = encoder(pos.to(device))
                out_pos.append(out)
            
            loss = AttentioNCE(out_1, out_2, out_pos, batch_size=n_anchors)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Perform evaluation every n_eval_epochs
        if (epoch + 1) % n_eval_epochs == 0:
            acc = knn_evaluation(encoder, dataset='cifar10')
            accs.append(acc)
            print(f"Epoch {epoch + 1}: KNN Evaluation Accuracy: {acc:.2f}")
        
        # Save model and optimizer state every n_save_epochs
        if (epoch + 1) % n_save_epochs == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "accs": accs
            }

            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_pos', type=int, default=4, help='Number of positives.')
    parser.add_argument('--projection_dim', type=int, default=128, help='Dimension of the projection layer.')
    parser.add_argument('--n_anchors', type=int, default=128, help='Number of anchors (default: 128).')
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs.')

    parser.add_argument('--save_dir', type=str, default="checkpointschanged_model_transfom", 
                        help='Directory to save checkpoints.')
    parser.add_argument('--n_eval_epochs', type=int, default=5, 
                        help='Frequency of evaluation in epochs.')
    parser.add_argument('--n_save_epochs', type=int, default=20, 
                        help='Frequency of saving checkpoints in epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay.')

    args = parser.parse_args()
    print(args)
    
    main(args)