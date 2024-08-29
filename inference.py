import torch
import torchvision.transforms as transforms
from PIL import Image
from src.model import CNNModel  
import os

def load_model(model_path, device):
    model = CNNModel(num_classes=10)  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  
        transforms.Grayscale(),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  
    ])
    image = Image.open(image_path).convert('RGB')  
    image = transform(image)
    image = image.unsqueeze(0) 
    return image

def predict_image(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)  
        return pred.item()

if __name__ == "__main__":
    model_path = 'model/simple_cnn.pth' 
    # image_path = 'external_Test/513ad2b0cfad8ad3dce41e82ad5150c4_t.jpeg' 
    # image_path='external_Test/CF1ze.jpg'
    # image_path='external_Test/images.png'
    # image_path='external_Test/0_kKxxK1YXSyWMEBtS.png'
    # image_path = 'external_Test/5.png' 
    image_path = 'external_Test/images7.png'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
    elif not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
    else:
        model = load_model(model_path, device)
        image = preprocess_image(image_path)
        prediction = predict_image(model, image, device)
        print(f'Predicted class: {prediction}')
