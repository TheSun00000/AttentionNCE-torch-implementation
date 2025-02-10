#imports
import torch
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import numpy as np


def get_knn_evaluation_loader(dataset_name, batch_size=512):
    
    if dataset_name in ['cifar10', 'svhn', 'cifar100']:
        single_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        if dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10('./dataset/cifar10/', train=True, transform=single_transform)
            test_dataset = torchvision.datasets.CIFAR10('./dataset/cifar10/', train=False, transform=single_transform)
        elif dataset_name == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100('./dataset/cifar100', train=True, transform=single_transform)
            test_dataset = torchvision.datasets.CIFAR100('./dataset/cifar100', train=False, transform=single_transform)
        elif dataset_name == 'svhn':
            train_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='train', transform=single_transform)
            test_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='test', transform=single_transform)
    
    
    elif dataset_name in ['TinyImagenet', 'stl10']:
        single_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if dataset_name == 'TinyImagenet':
            train_dataset = torchvision.datasets.ImageFolder('./dataset/tiny-imagenet-200/train', transform=single_transform)
            test_dataset = torchvision.datasets.ImageFolder('./dataset/tiny-imagenet-200/val/images', transform=single_transform)
        elif dataset_name == 'stl10': 
            train_dataset = torchvision.datasets.STL10('./dataset/STL10/', split='train', transform=single_transform)
            test_dataset = torchvision.datasets.STL10('./dataset/STL10/', split='test', transform=single_transform)
            
    memory_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )
        
    return memory_loader, test_loader



def knn_monitor(net, memory_data_loader, test_data_loader, device='cuda', k=200, t=0.1, targets=None):
    if not targets:
        if 'targets' in memory_data_loader.dataset.__dir__():
            targets = memory_data_loader.dataset.targets
        elif 'labels' in memory_data_loader.dataset.__dir__():
            targets = memory_data_loader.dataset.labels
        else:
            raise NotImplementedError
        
        
    net.eval()
    classes = len(np.unique(targets))
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        # for data, target in tqdm(memory_data_loader, total=len(memory_data_loader), desc='memory_data_loader'):
        for data, target in memory_data_loader:
            feature = net(data.to(device=device, non_blocking=True)).reshape(len(data), -1)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device, dtype=torch.int64)
        # loop test data to predict the label by weighted knn search
        # for data, target in tqdm(test_data_loader, total=len(test_data_loader), desc='test_data_loader'):
        for data, target in test_data_loader:
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            feature = net(data).reshape(len(data), -1)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)
            
            total_num += data.size(0)

            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """
    feature and feature_bank are normalized
    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()
    
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # print(one_hot_label)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels         
    

def knn_evaluation(encoder, dataset):
    
    memory_loader, test_loader = get_knn_evaluation_loader(dataset, batch_size=512)
    
    acc = knn_monitor(
        encoder.encoder,
        memory_loader,
        test_loader,
    )
    
    return acc