
import torch
import numpy as np
from tqdm import tqdm

from random import sample
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np



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



# Function to generate embeddings for test data
def generate_embeddings(model, data_loader,device):
    model.eval()
    embeddings, labels = [], []

    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Generating Embeddings"):
            data = data.to(device)
            # Extract features from the encoder
            feature = model.encoder(data)  # Forward pass only through the encoder
            embeddings.append(feature.cpu().numpy())
            labels.append(target.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels

# Apply t-SNE to reduce embeddings to 2D for visualization
def visualize_tsne(embeddings, labels, num_classes):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))  # Use a colormap for distinct colors

    for class_idx in range(num_classes):
        class_points = reduced_embeddings[labels == class_idx]
        plt.scatter(class_points[:, 0], class_points[:, 1], label=f'Class {class_idx}', alpha=0.7, color=colors[class_idx])

    plt.title('t-SNE Visualization of Test Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    


