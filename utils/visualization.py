import torch
import numpy as np
import matplotlib.pyplot as plt
from data import load_data



def visualize_batch(loader):
    batch = next(iter(loader))  # Get the first batch
    images, labels = batch
    
    fig, axes = plt.subplots(1, 6, figsize=(15, 4))
    for i, (image, label) in enumerate(zip(images[:6], labels[:6])):
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {label.item()}")
        axes[i].axis('off')
    plt.show()

# loss plot and average loss accuracy
