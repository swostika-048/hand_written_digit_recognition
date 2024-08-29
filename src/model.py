import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        
        self.model = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second Convolutional Block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Flatten Layer
            nn.Flatten(),
            
            # Fully Connected Layer
            nn.Linear(64 * 7 * 7, 128),  # Adjust input size based on the output of the previous layers
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Output Layer
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# # Example usage
# model = CNNModel(num_classes=10)
# print(model)
