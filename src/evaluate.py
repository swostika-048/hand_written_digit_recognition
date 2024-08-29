import torch
from data_loader import create_dataloaders
from model import CNNModel
from data import load_data

def evaluate_model(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy

if __name__ == "__main__":
    X, y = load_data('data/mnist-original.mat')
    _, test_loader = create_dataloaders(X, y)
    model = CNNModel(10) 
    evaluate_model(test_loader, model)
