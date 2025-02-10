import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from knn_evaluation import knn_evaluation
from resnet import Model as ResNet18

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

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


class ResNet18Encoder(nn.Module):
    def __init__(self, projection_dim):
        super(ResNet18Encoder, self).__init__()
        # Load pre-trained ResNet18 and remove the fully connected layer
        self.encoder = ResNet18()
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        # Add a projection head
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        # Forward pass through ResNet18 encoder
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)  # Flatten the output
        # Forward pass through the projection head
        projections = self.projection_head(features)
        projections = nn.functional.normalize(projections, p=2, dim=1)
        
        return projections


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, symmetric=True):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # Create the encoders
        self.encoder_q = ResNet18Encoder(projection_dim=dim)
        self.encoder_k = ResNet18Encoder(projection_dim=dim)

        # Initialize the key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Don't update by gradient

        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # Transpose
        ptr = (ptr + batch_size) % self.K  # Move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """Batch shuffle for making use of BatchNorm."""
        idx_shuffle = torch.randperm(x.shape[0]).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """Undo batch shuffle."""
        return x[idx_unshuffle]
    
    def AttentionCE(self, anchor, out_pos, batch_size, temperature=0.5, d_pos=1., d_neg=1.):
        """
        Contrastive loss computation.
        """
        feature_dim = 128
        num_pos = len(out_pos)
        
        # Positive score calculation
        pos_views = torch.cat(out_pos, dim=1).view(-1, num_pos, feature_dim)
        pos_sim = torch.sum(anchor.view(1, batch_size, 1, feature_dim) * pos_views, dim=-1).view(-1, num_pos)
        alpha = torch.nn.functional.softmax(pos_sim.detach() / d_pos, dim=-1)
        pos_score = torch.exp(torch.sum(pos_sim * alpha, dim=-1) / temperature)

        # Negative score calculation
        neg_sim = torch.mm(anchor, self.queue.clone().detach())
        beta = (torch.nn.functional.softmax(neg_sim.detach() / d_neg, dim=-1) + 1e-6) * (self.queue.shape[0])
        neg_score = torch.exp(neg_sim * beta / temperature)

        # Contrastive loss
        loss = -torch.log(pos_score / (pos_score + neg_score.sum(dim=-1))).mean()
        return loss

    def forward(self, im1, positives):
        """
        Forward pass through the model.
        """
        # Update the key encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
        q = self.encoder_q(im1)
        q = nn.functional.normalize(q, dim=1)
        
        out_pos_k = []
        with torch.no_grad():
            for pos in positives:
                k = self.encoder_k(pos)
                k = nn.functional.normalize(k, dim=1)
                out_pos_k.append(k)

        # Calculate contrastive loss
        loss = self.AttentionCE(
            anchor=q,
            out_pos=out_pos_k,
            batch_size=q.shape[0]
        )

        for k in out_pos_k:
            self._dequeue_and_enqueue(k)

        return loss


# Hyperparameters and data setup
n_pos = 3
projection_dim = 128
n_anchors = 256
epochs = 400

# Dataset and DataLoader setup
cifar10_train = datasets.CIFAR10(root="./dataset/cifar10", train=True, download=True)
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])
simclr_dataset = SimCLRDataset(cifar10_train, transform, n_pos)
data_loader = DataLoader(simclr_dataset, batch_size=n_anchors, shuffle=True, drop_last=True)

# Model, optimizer, and training setup
moco_model = ModelMoCo().to(device)
optimizer = torch.optim.Adam(moco_model.parameters(), lr=1e-3)

losses = []
accs = []

acc = None
# Training loop
for epoch in range(epochs):
    tqdm_data_loader = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')
    for n_step, (anchors, positives, labels) in tqdm_data_loader:
        
        anchor1, anchor2 = anchors
        anchor1, anchor2 = anchors[0].to(device), anchors[1].to(device)
        positives = [pos.to(device) for pos in positives]
        
        loss = moco_model(anchor1, positives)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        # Plot training loss and accuracy
        if n_step % (len(data_loader) // 10) == 0 or n_step == len(data_loader)-1:
            tqdm_data_loader.set_description(f'Epoch {epoch}\tLoss: {loss.item():.4f}\tAcc: {acc}')

    acc = knn_evaluation(moco_model.encoder_q, dataset='cifar10')
    acc = round(acc, 2)