import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from data import load_data
# from utils.visualization import visualize_batch
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].unsqueeze(0), self.labels[idx]  # Add channel dimension

def create_dataloaders(X, y, batch_size=32):
    dataset = MNISTDataset(X, y)
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



# if __name__ == "__main__":
#     X, y = load_data('data/mnist-original.mat')

#     train_loader, test_loader = create_dataloaders(X, y)
#     print(f"Number of batches in training set: {len(train_loader)}")
#     print(f"Number of batches in test set: {len(test_loader)}")

   
#     visualize_batch(train_loader)
