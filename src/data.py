import scipy.io
import numpy as np

import matplotlib.pyplot as plt

def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    
    
    images = data['data'] 
    images = images.T 
    images = images.reshape(-1, 28, 28).astype(np.float32) / 255.0 
    
    
    labels = data['label']  
    labels = np.squeeze(labels).astype(np.int64) 
    
    return images, labels



def visualize_images(images, labels, num_images=2):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[labels[2]], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    x, y = load_data('data/mnist-original.mat')
    visualize_images(x, y)


