import scipy.io
import numpy as np
# Visualization Function
import matplotlib.pyplot as plt

def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    
    # Extract and reshape the image data
    images = data['data']  # Shape: (784, 70000)
    images = images.T  # Transpose to shape (70000, 784)
    images = images.reshape(-1, 28, 28).astype(np.float32) / 255.0  # Reshape to (70000, 28, 28) and normalize
    
    # Extract and reshape the label data
    labels = data['label']  # Shape: (1, 70000)
    labels = np.squeeze(labels).astype(np.int64)  # Squeeze to shape (70000,)
    
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


