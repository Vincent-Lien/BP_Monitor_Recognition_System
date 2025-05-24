import torch
from PIL import Image
from torchvision.models import resnet18

import torch.nn as nn
import torchvision.transforms as transforms

def load_model(model_path, device):
    """
    Load the trained seven segment classifier model
    
    Args:
        model_path (str): Path to the saved model weights
        device (torch.device): Device to run the model on
        
    Returns:
        model (torch.nn.Module): Loaded model
    """
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image):
    """
    Preprocess an image for inference
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tensor (torch.Tensor): Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.fromarray(image).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension

def inference_single_image(image_path):
    """
    Run inference on a single image
    
    Args:
        model (torch.nn.Module): Loaded model
        image_path (str): Path to the image file
        device (torch.device): Device to run inference on
        
    Returns:
        pred (int): Predicted digit
        confidence (float): Confidence score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model('checkpoints/seven_seg_classification_best.pth', device)

    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    return prediction.item(), confidence.item()